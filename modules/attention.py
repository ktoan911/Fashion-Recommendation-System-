import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalAttention(nn.Module):
    """
    nhân trọng số quan trọng của feature map
    """

    def __init__(self, in_dim):
        super().__init__()
        # Convolution 1x1
        out_dim = 1
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x_spatial):
        # x_spatial: [B, C, H, W]
        # attention map
        attn_map = self.conv(x_spatial)  # [B, 1, H, W]

        attn_map = F.softmax(
            attn_map.view(x_spatial.size(0), -1),  # [B, H*W]
            dim=1,
        ).view(x_spatial.size(0), 1, x_spatial.size(2), x_spatial.size(3))
        # (B, C, H, W) * (B, 1, H, W) -> (B, C, H, W) -> sum ọn H, W -> (B, C)
        x_global = (x_spatial * attn_map).sum(dim=(2, 3))
        # x_global: [B, C]
        return x_global


class LandmarkAttention(nn.Module):
    """
    Kết hợp các đặc trưng landmark (L x d_img) thành một vector toàn cục (d_img).
    Được mô tả là một convolution 1x1. Tương đương với một lớp Linear.
    """

    def __init__(self, d_img):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_img, d_img // 2), nn.ReLU(), nn.Linear(d_img // 2, 1)
        )

    def forward(self, x_landmarks):
        # x_landmarks: [B, L, C]  # B: batch size, L: số lượng landmarks, C: số chiều đặc trưng
        weights = self.attn(x_landmarks)  # -> [B, L, 1]
        weights = F.softmax(weights, dim=1)

        # Tổng hợp có trọng số
        x_global = (x_landmarks * weights).sum(dim=1)  # -> [B, C]
        # x_global: [B, C]
        return x_global


class ContextualAttention(nn.Module):

    def __init__(self, d_img, d_fused_out):
        super().__init__()
        self.fuse_layer = nn.Sequential(
            nn.Linear(d_img * 3, d_fused_out), nn.ReLU(), nn.LayerNorm(d_fused_out)
        )

    def forward(self, f_whole, f_crop, f_lmk):
        # f_whole: [B, C], f_crop: [B, C], f_lmk: [B, C]
        combined_features = torch.cat([f_whole, f_crop, f_lmk], dim=1)  # -> [B, C*3]
        fused_rep = self.fuse_layer(combined_features)
        # fused_rep: [B, d_fused_out]
        return fused_rep
