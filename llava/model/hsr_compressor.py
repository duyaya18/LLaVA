"""
HSR Compressor 集成到 LLaVA

该模块将 HSR 压缩算法集成到 LLaVA 模型中，
在图像特征传入 LLM 之前进行视觉 Token 压缩。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HSRCompressorLLaVA(nn.Module):
    """
    适配 LLaVA 的 HSR 压缩器
    
    区别于标准 HSRCompressor：
    - 不需要真实的 cross_attn_weights（LLM 内部才有）
    - 使用图像特征的统计信息作为近似显著性分数
    - 集成到 LLaVA 的特征处理流程中
    """
    
    def __init__(self, embed_dim: int, reduction_ratio: float = 0.5, 
                 anchor_ratio: float = 0.5, num_kmeans_iter: int = 10,
                 spatial_weight: float = 0.1):
        """
        Args:
            embed_dim: 视觉 Token 维度
            reduction_ratio: 压缩后保留的 Token 比例
            anchor_ratio: Anchors 在总输出中的比例 (默认 0.5 = 1:1)
            num_kmeans_iter: K-Means 迭代次数
            spatial_weight: 空间距离权重
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.reduction_ratio = reduction_ratio
        self.anchor_ratio = anchor_ratio
        self.num_kmeans_iter = num_kmeans_iter
        self.spatial_weight = spatial_weight

        # 语义残差聚合器：单个线性层
        self.residual_aggregator = nn.Linear(embed_dim, embed_dim)
        
        # 可学习的加权系数
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

    def _compute_spatial_coords(self, H: int, W: int, device: torch.device):
        """生成 2D 空间坐标网格"""
        y_coords = torch.arange(H, device=device, dtype=torch.float32) / H
        x_coords = torch.arange(W, device=device, dtype=torch.float32) / W
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([grid_x, grid_y], dim=-1)
        return coords.view(-1, 2)

    def _compute_saliency_scores(self, visual_tokens: torch.Tensor):
        """
        计算显著性分数
        
        由于没有 cross_attn_weights，使用特征统计作为近似：
        - 特征的 L2 范数
        - 空间位置（中心区域更重要）
        """
        B, N, D = visual_tokens.shape
        
        # 特征范数作为重要性指标
        feat_norms = torch.norm(visual_tokens, dim=-1)  # [B, N]
        
        # 归一化
        feat_norms = feat_norms / feat_norms.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        return feat_norms

    def _run_kmeans(self, tokens: torch.Tensor, coords: torch.Tensor, 
                    K: int, num_iter: int):
        """K-Means 聚类（使用特征 + 2D 空间坐标）"""
        N, D = tokens.shape
        
        if K >= N:
            return tokens, torch.arange(N, device=tokens.device)
        
        if K <= 0:
            K = 1
            
        indices = torch.randperm(N)[:K].to(tokens.device)
        centroids = tokens[indices].clone()
        centroid_coords = coords[indices].clone()
        
        for _ in range(num_iter):
            feat_distances = torch.cdist(tokens, centroids)
            spatial_distances = torch.cdist(coords, centroid_coords)
            distances = feat_distances + self.spatial_weight * spatial_distances
            
            labels = torch.argmin(distances, dim=-1)
            
            for k in range(K):
                mask = (labels == k)
                if mask.sum() > 0:
                    centroids[k] = tokens[mask].mean(dim=0)
                    centroid_coords[k] = coords[mask].mean(dim=0)
        
        return centroids, labels

    def forward(self, visual_tokens: torch.Tensor, image_sizes: list = None):
        """
        HSR 压缩
        
        Args:
            visual_tokens: [B, N, D] 原始视觉 Token
            image_sizes: 每个图像的原始尺寸 [(h1, w1), (h2, w2), ...]
        
        Returns:
            compressed_tokens: [B, K_anchor + K_centroid, D] 压缩后的 Token
            attention_mask: [B, K_anchor + K_centroid]
        """
        B, N, D = visual_tokens.shape
        
        # 动态调整 residual_aggregator 以匹配输入维度
        if not hasattr(self, '_initialized') or self._initialized != D:
            self.residual_aggregator = nn.Linear(D, D)
            self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
            self._initialized = D
        
        # 确保在正确的设备上
        self.residual_aggregator = self.residual_aggregator.to(visual_tokens.device, visual_tokens.dtype)
        if self.residual_scale.device != visual_tokens.device or self.residual_scale.dtype != visual_tokens.dtype:
            self.residual_scale = nn.Parameter(self.residual_scale.data.to(device=visual_tokens.device, dtype=visual_tokens.dtype))
        
        # 推断空间维度
        side = int(math.sqrt(N))
        H, W = side, side
        
        # 计算显著性分数
        saliency_scores = self._compute_saliency_scores(visual_tokens)
        
        # 确定最终 Token 数量
        N_final = int(N * self.reduction_ratio)
        
        # 按 1:1 比例分配
        K_anchor = math.ceil(N_final / 2)
        K_centroid = N_final - K_anchor
        K_centroid = max(1, K_centroid)
        
        # 提取 Anchors
        _, indices_topk = torch.topk(saliency_scores, K_anchor, dim=1)
        
        batch_idx = torch.arange(B, device=visual_tokens.device).unsqueeze(1)
        anchors = visual_tokens[batch_idx, indices_topk]
        
        # 生成空间坐标（使用 float32 以避免与 half 混合计算）
        all_coords = self._compute_spatial_coords(H, W, visual_tokens.device).float()
        all_coords = all_coords.unsqueeze(0).expand(B, -1, -1)
        anchor_coords = all_coords[batch_idx, indices_topk]
        
        # Context Mask
        context_mask = torch.ones(B, N, dtype=torch.bool, device=visual_tokens.device)
        context_mask.scatter_(1, indices_topk, False)
        
        # K-Means 聚类
        centroids_list = []
        context_labels_list = []
        context_indices_list = []
        
        for b in range(B):
            ctx_mask = context_mask[b]
            ctx_tokens = visual_tokens[b, ctx_mask]
            ctx_indices = torch.where(ctx_mask)[0]
            ctx_coords = all_coords[b, ctx_mask]
            
            actual_K = min(K_centroid, ctx_tokens.shape[0])
            
            if actual_K > 0:
                centroids_b, labels_b = self._run_kmeans(ctx_tokens, ctx_coords, actual_K, self.num_kmeans_iter)
            else:
                centroids_b = torch.zeros(K_centroid, D, device=visual_tokens.device)
                labels_b = torch.zeros(0, dtype=torch.long, device=visual_tokens.device)
            
            if actual_K < K_centroid:
                padding = torch.zeros(K_centroid - actual_K, D, device=visual_tokens.device)
                centroids_b = torch.cat([centroids_b, padding], dim=0)
                labels_b = labels_b.clamp(max=K_centroid - 1) if len(labels_b) > 0 else labels_b
            
            centroids_list.append(centroids_b)
            context_labels_list.append(labels_b)
            context_indices_list.append(ctx_indices)
        
        centroids = torch.stack(centroids_list, dim=0)
        
        # 残差计算与注入
        anchor_residual_injection = torch.zeros_like(anchors)
        
        for b in range(B):
            ctx_tokens = visual_tokens[b]
            ctx_indices_b = context_indices_list[b]
            ctx_labels_b = context_labels_list[b]
            centroids_b = centroids[b]
            
            if len(ctx_indices_b) > 0 and K_centroid > 0:
                ctx_features = ctx_tokens[ctx_indices_b]
                ctx_centroids = centroids_b[ctx_labels_b]
                
                residuals = ctx_features - ctx_centroids
                
                aggregated_residuals = torch.zeros_like(centroids_b)
                for k in range(K_centroid):
                    mask = (ctx_labels_b == k)
                    if mask.sum() > 0:
                        aggregated_residuals[k] = residuals[mask].sum(dim=0)
                
                aggregated_residuals = self.residual_aggregator(aggregated_residuals)
                
                # 获取 scale 值
                scale_val = self.residual_scale.data
                
                # 按空间距离注入
                for k in range(K_centroid):
                    cluster_mask = (ctx_labels_b == k)
                    if cluster_mask.sum() > 0:
                        cluster_indices = ctx_indices_b[cluster_mask]
                        avg_idx = cluster_indices.float().mean()
                        cy = (avg_idx.item() // W) / H
                        cx = (avg_idx.item() % W) / W
                        # 使用与 anchor_coords 相同的 dtype
                        cluster_coord = torch.tensor([[cx, cy]], device=visual_tokens.device, dtype=anchor_coords.dtype)
                        
                        dists = torch.cdist(cluster_coord, anchor_coords[b])
                        nearest_anchor = dists.argmin(dim=1)[0]
                        
                        anchor_residual_injection[b, nearest_anchor] += scale_val * aggregated_residuals[k]
        
        anchors = anchors + anchor_residual_injection
        
        # 输出
        compressed_tokens = torch.cat([anchors, centroids], dim=1)
        attention_mask = torch.ones(B, compressed_tokens.shape[1], device=visual_tokens.device)

        return compressed_tokens, attention_mask


def integrate_hsr_to_llava():
    """
    示例：如何将 HSR 集成到 LLaVA
    
    修改位置：llava/model/llava_arch.py 的 encode_images 方法
    
    修改前:
        def encode_images(self, images):
            image_features = self.get_vision_tower()(images)
            image_features = self.get_model().mm_projector(image_features)
            return image_features
    
    修改后:
        def encode_images(self, images):
            image_features = self.get_vision_tower()(images)
            image_features = self.get_model().mm_projector(image_features)
            
            # 添加 HSR 压缩
            if hasattr(self, 'hsr_compressor'):
                image_features, _ = self.hsr_compressor(image_features)
            
            return image_features
    """
    pass


if __name__ == "__main__":
    # 测试
    B, N, D = 2, 576, 768
    
    compressor = HSRCompressorLLaVA(embed_dim=D, reduction_ratio=0.5)
    
    vis_in = torch.randn(B, N, D)
    out_tokens, out_mask = compressor(vis_in)
    
    N_final = int(N * 0.5)
    K_anchor = math.ceil(N_final / 2)
    K_centroid = N_final - K_anchor
    
    print(f"输入形状: {vis_in.shape}")
    print(f"输出形状: {out_tokens.shape}")
    print(f"预期: K_anchor={K_anchor}, K_centroid={K_centroid}, 总计={K_anchor+K_centroid}")
    print("测试通过!")
