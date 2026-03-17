"""
HSR LLaVA 推理测试脚本

使用方法:
    # 启用 HSR 压缩:
    python test_hsr_inference.py --model-path <模型路径> --image-file test.png --prompt "描述这个图像" --enable-hsr
    
    # 不启用 HSR 压缩 (对比):
    python test_hsr_inference.py --model-path <模型路径> --image-file test.png --prompt "描述这个图像"

示例:
    python test_hsr_inference.py \
        --model-path liuhaotian/llava-v1.5-7b \
        --image-file test.png \
        --prompt "描述这个图像" \
        --enable-hsr \
        --hsr-reduction-ratio 0.5
"""

import argparse
import math
import time
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model, enable_hsr_compression
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, 
        args.load_8bit, args.load_4bit, device=args.device
    )

    # 启用 HSR 压缩
    if args.enable_hsr:
        # 获取视觉特征的维度：从 mm_projector 输出获取
        mm_hidden_size = None
        
        if hasattr(model.get_model(), 'mm_projector'):
            projector = model.get_model().mm_projector
            # 尝试获取 projector 输出维度
            if hasattr(projector, 'out_features'):
                mm_hidden_size = projector.out_features
            elif hasattr(projector, 'config'):
                mm_hidden_size = getattr(projector.config, 'hidden_size', None)
        
        # 如果还没获取到，尝试从 vision tower
        if mm_hidden_size is None:
            vision_tower = model.get_vision_tower()
            if vision_tower is not None:
                mm_hidden_size = vision_tower.hidden_size
        
        # 最后使用默认值
        if mm_hidden_size is None:
            mm_hidden_size = 4096  # LLaVA-1.5-7b 的 LLM 隐藏层大小
        
        print(f"[DEBUG] 视觉特征维度: {mm_hidden_size}")
        
        model = enable_hsr_compression(
            model,
            embed_dim=mm_hidden_size,
            reduction_ratio=args.hsr_reduction_ratio,
            anchor_ratio=args.hsr_anchor_ratio,
            num_kmeans_iter=args.hsr_kmeans_iter,
            spatial_weight=args.hsr_spatial_weight
        )
        
        # 将 HSR 压缩器移到模型设备上
        if hasattr(model, 'get_model') and hasattr(model.get_model(), 'hsr_compressor'):
            hsr = model.get_model().hsr_compressor
            if hsr is not None:
                hsr = hsr.to(model.device)
                model.get_model().hsr_compressor = hsr

    # 选择对话模式
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None:
        conv_mode = args.conv_mode

    conv = conv_templates[conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    # 加载图像
    image = load_image(args.image_file)
    image_size = image.size
    image_tensor = process_images([image], image_processor, model.config)
    
    if type(image_tensor) is list:
        image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    print(f"[INFO] 模型: {args.model_path}")
    print(f"[INFO] 图像文件: {args.image_file}")
    print(f"[INFO] 图像尺寸: {image_size}")
    print(f"[INFO] Prompt: {args.prompt}")

    # 打印 HSR 状态
    hsr_active = False
    if hasattr(model, 'get_model') and hasattr(model.get_model(), 'hsr_compressor'):
        hsr = model.get_model().hsr_compressor
        if hsr is not None:
            hsr_active = True
            print(f"[INFO] HSR 压缩: 已启用")
            print(f"       - reduction_ratio: {hsr.reduction_ratio}")
            print(f"       - anchor_ratio: {hsr.anchor_ratio}")
            print(f"       - K-Means 迭代: {hsr.num_kmeans_iter}")
            print(f"       - 空间权重: {hsr.spatial_weight}")
    
    if not hsr_active:
        print(f"[INFO] HSR 压缩: 未启用 (使用原始视觉特征)")

    # 构建提示
    inp = args.prompt
    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # 生成
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    
    # 统计输入 token 数量
    input_token_count = input_ids.shape[1]
    
    # 打印 HSR 压缩信息
    hsr_enabled = False
    original_vision_tokens = 576  # LLaVA-1.5 默认值
    compressed_vision_tokens = None
    
    if hasattr(model, 'get_model') and hasattr(model.get_model(), 'hsr_compressor'):
        hsr = model.get_model().hsr_compressor
        if hsr is not None:
            hsr_enabled = True
            # 计算压缩后的 token 数量
            N = original_vision_tokens
            N_final = int(N * hsr.reduction_ratio)
            K_anchor = math.ceil(N_final / 2)
            K_centroid = N_final - K_anchor
            compressed_vision_tokens = K_anchor + K_centroid
    
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print(f"\n[{roles[1]}]: ", end="", flush=True)
    
    start_time = time.time()
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            use_cache=True
        )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    output_text = tokenizer.decode(output_ids[0]).strip()
    output_token_count = output_ids.shape[1]
    
    # 打印统计信息
    print(f"\n\n" + "="*50)
    print("========== 统计信息 ==========")
    print("="*50)
    print(f"输入 Token 数量: {input_token_count}")
    print(f"输出 Token 数量: {output_token_count}")
    if original_vision_tokens is not None:
        print(f"视觉 Token (压缩前): {original_vision_tokens}")
    if compressed_vision_tokens is not None:
        print(f"视觉 Token (压缩后): {compressed_vision_tokens}")
        compression_ratio = (1 - compressed_vision_tokens / original_vision_tokens) * 100 if original_vision_tokens else 0
        print(f"压缩率: {compression_ratio:.1f}%")
    print(f"生成耗时: {elapsed_time:.2f} 秒")
    print(f"生成速度: {output_token_count / elapsed_time:.2f} tokens/秒")
    print("="*50)
    print("========== 完整输出 ==========")
    print(output_text)
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="模型路径或 HuggingFace 模型 ID")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default="test.png")
    parser.add_argument("--prompt", type=str, default="描述这个图像")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    
    # HSR 参数
    parser.add_argument("--enable-hsr", action="store_true", help="启用 HSR 压缩")
    parser.add_argument("--hsr-reduction-ratio", type=float, default=0.5, help="HSR 压缩比例")
    parser.add_argument("--hsr-anchor-ratio", type=float, default=0.5, help="HSR anchor 比例 (1:1)")
    parser.add_argument("--hsr-kmeans-iter", type=int, default=10, help="K-Means 迭代次数")
    parser.add_argument("--hsr-spatial-weight", type=float, default=0.1, help="空间距离权重")
    
    args = parser.parse_args()
    main(args)
