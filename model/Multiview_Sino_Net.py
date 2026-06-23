"""
@file: Sino_Net.py
@brief: 多角度统一正弦域补全网络
@details:
    用于 full view = 36，target view = 6~18 的正弦图补全任务。

    输入:
        x: [B, 3, 367, 36]
           x[:, 0:1] = sparse_sino
           x[:, 1:2] = mask
           x[:, 2:3] = view_ratio_map

    输出:
        S_clean: [B, 1, 367, 36]
        U_sino : [B, 1, 367, 36]

    说明:
        1. S_clean 用于预测完整正弦图。
        2. U_sino 是不确定性图，后续可用于图像域融合。
        3. 默认 keep_known=False，让网络自行学习已知角度处的校正。
        4. loss 中显式加入 known consistency，避免已知角度被改坏。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# LayerNorm2d
# =========================================================
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps

        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)

        y = (x - mu) / (var + eps).sqrt()

        ctx.save_for_backward(y, var, weight)

        y = weight.view(1, -1, 1, 1) * y + bias.view(1, -1, 1, 1)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        y, var, weight = ctx.saved_tensors

        g = grad_output * weight.view(1, -1, 1, 1)

        mean_g = g.mean(1, keepdim=True)
        mean_gy = (g * y).mean(1, keepdim=True)

        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)

        grad_weight = (grad_output * y).sum((0, 2, 3))
        grad_bias = grad_output.sum((0, 2, 3))

        return gx, grad_weight, grad_bias, None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


# =========================================================
# Attention Modules
# =========================================================
class CAB(nn.Module):
    """
    Channel Attention Block
    """
    def __init__(self, in_channels, ratio=16):
        super().__init__()

        hidden_channels = max(1, in_channels // ratio)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Block
    """
    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in [3, 7], "kernel_size should be 3 or 7"

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn)

        return self.sigmoid(attn)


class SimpleGate(nn.Module):
    """
    NAFNet 风格 SimpleGate
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# =========================================================
# MDPR Block
# =========================================================
class MDPRBlock(nn.Module):
    """
    Multi-Dilation Perception Residual Block
    """
    def __init__(self, c, norm_groups=8):
        super().__init__()

        if c % norm_groups != 0:
            raise ValueError(
                f"channels c={c} must be divisible by norm_groups={norm_groups}"
            )

        self.norm1 = nn.GroupNorm(norm_groups, c)

        self.conv1 = nn.Conv2d(c, c * 2, kernel_size=1)

        self.conv2 = nn.Conv2d(
            c * 2,
            c * 2,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=c * 2,
        )

        self.sg = SimpleGate()

        self.conv_big = nn.Conv2d(
            c,
            c,
            kernel_size=7,
            padding=3,
            groups=c,
        )

        self.ca = CAB(c)
        self.sa = SpatialAttention(kernel_size=7)

        self.conv3 = nn.Conv2d(c, c, kernel_size=1)

        self.norm2 = nn.GroupNorm(norm_groups, c)

        self.conv4 = nn.Conv2d(c, c * 2, kernel_size=1)
        self.conv5 = nn.Conv2d(c, c, kernel_size=1)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.sg(out)

        out = self.conv_big(out)

        out = out * self.ca(out)
        out = out * self.sa(out)

        out = self.conv3(out)

        x = residual + out * self.beta

        residual = x

        out = self.norm2(x)
        out = self.conv4(out)
        out = self.sg(out)
        out = self.conv5(out)

        x = residual + out * self.gamma

        return x


# =========================================================
# Multi-view Sino Domain Network
# =========================================================
class MDPR_SinoDomain(nn.Module):
    """
    多角度统一正弦域补全网络。

    适用场景:
        full view = 36
        target view = 6~18
        detector = 367

    输入:
        x: [B, 3, 367, 36]

    输出:
        S_clean: [B, 1, 367, 36]
        U_sino : [B, 1, 367, 36]
    """
    def __init__(
        self,
        in_channels=3,
        width=32,
        num_blocks=6,
        keep_known=False,
        norm_groups=8,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.width = width
        self.num_blocks = num_blocks
        self.keep_known = keep_known

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            *[
                MDPRBlock(width, norm_groups=norm_groups)
                for _ in range(num_blocks)
            ]
        )

        self.clean_head = nn.Conv2d(width, 1, kernel_size=3, padding=1)

        self.uncert_head = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 1, kernel_size=3, padding=1),
        )

        self.uncert_act = nn.Sigmoid()

    def forward(self, x, return_dict=False, clamp_output=False):
        """
        Args:
            x:
                [B, 3, H, W]
                channel 0: sparse_sino
                channel 1: mask
                channel 2: view_ratio_map

            return_dict:
                是否返回 dict。

            clamp_output:
                是否把 S_clean 限制到 [0,1]。
                训练时建议 False。
                验证/测试/保存图像时可以 True。

        Returns:
            S_clean, U_sino
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected x shape [B, C, H, W], got {tuple(x.shape)}"
            )

        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.size(1)}"
            )

        sparse_sino = x[:, 0:1, :, :]
        mask = x[:, 1:2, :, :]

        feat = self.input_proj(x)
        feat = self.blocks(feat)

        residual = self.clean_head(feat)

        S_clean = sparse_sino + residual

        if self.keep_known:
            S_clean = S_clean * (1.0 - mask) + sparse_sino * mask

        if clamp_output:
            S_clean = torch.clamp(S_clean, 0.0, 1.0)

        U_sino = self.uncert_head(feat)
        U_sino = self.uncert_act(U_sino)

        if return_dict:
            return {
                "S_clean": S_clean,
                "U_sino": U_sino,
            }

        return S_clean, U_sino


# =========================================================
# Loss Functions
# =========================================================
def masked_l1_loss(pred, target, weight_mask, eps=1e-6):
    """
    只在指定 mask 区域计算 L1。

    Args:
        pred:
            [B, 1, H, W]
        target:
            [B, 1, H, W]
        weight_mask:
            [B, 1, H, W]
            需要计算 loss 的区域为 1，其余为 0。
    """
    if weight_mask.dim() == 3:
        weight_mask = weight_mask.unsqueeze(1)

    loss = torch.abs(pred - target) * weight_mask
    return loss.sum() / (weight_mask.sum() + eps)


def sino_completion_loss(
    pred,
    target,
    mask,
    full_weight=1.0,
    unknown_weight=2.0,
    known_weight=1.0,
):
    """
    正弦图补全损失，适用于 keep_known=False。

    设计目的:
        1. loss_full 保证整体正弦图接近 target。
        2. loss_unknown 重点优化缺失角度补全。
        3. loss_known 约束已知角度不要被网络改坏。

    Args:
        pred:
            [B, 1, 367, 36]

        target:
            [B, 1, 367, 36]

        mask:
            [B, 1, 367, 36]
            已知角度为 1，未知角度为 0。

        full_weight:
            整体 loss 权重。

        unknown_weight:
            未知角度区域 loss 权重。

        known_weight:
            已知角度区域 loss 权重。

    Returns:
        loss, loss_dict
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    unknown = 1.0 - mask

    loss_full = F.l1_loss(pred, target)
    loss_unknown = masked_l1_loss(pred, target, unknown)
    loss_known = masked_l1_loss(pred, target, mask)

    loss = (
        full_weight * loss_full
        + unknown_weight * loss_unknown
        + known_weight * loss_known
    )

    loss_dict = {
        "loss_full": loss_full.item(),
        "loss_unknown": loss_unknown.item(),
        "loss_known": loss_known.item(),
        "loss_total": loss.item(),
    }

    return loss, loss_dict


def uncertainty_supervision_loss(
    pred,
    target,
    U_sino,
    mask=None,
    detach_error=True,
):
    """
    可选的不确定性监督损失。

    说明:
        第一阶段可以先不用这个 loss。
        等 S_clean 正弦域补全稳定后，再加入:
            loss_total = loss_sino + 0.1 * loss_u

    Args:
        pred:
            [B, 1, H, W]

        target:
            [B, 1, H, W]

        U_sino:
            [B, 1, H, W]

        mask:
            [B, 1, H, W] or None
            如果传入 mask，则只监督未知角度区域的不确定性。

        detach_error:
            是否阻断 error_map 对 pred 的梯度。
    """
    error_map = torch.abs(pred - target)

    if detach_error:
        error_map = error_map.detach()

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        unknown = 1.0 - mask
        error_map = error_map * unknown

    max_val = error_map.amax(dim=(2, 3), keepdim=True)
    error_map = error_map / (max_val + 1e-6)

    loss_u = F.l1_loss(U_sino, error_map)

    return loss_u


# =========================================================
# Quick Test
# =========================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MDPR_SinoDomain(
        in_channels=3,
        width=32,
        num_blocks=6,
        keep_known=False,
        norm_groups=8,
    ).to(device)

    x = torch.rand(2, 3, 367, 36).to(device)

    # 模拟 sparse_sino
    x[:, 0:1, :, :] = torch.rand(2, 1, 367, 36).to(device)

    # 模拟 mask
    x[:, 1:2, :, :] = (torch.rand(2, 1, 367, 36).to(device) > 0.7).float()

    # 模拟 view_ratio_map
    x[:, 2:3, :, :] = 0.3333

    target = torch.rand(2, 1, 367, 36).to(device)
    mask = x[:, 1:2, :, :]

    S_clean, U_sino = model(x)

    loss, loss_dict = sino_completion_loss(
        pred=S_clean,
        target=target,
        mask=mask,
        full_weight=1.0,
        unknown_weight=2.0,
        known_weight=1.0,
    )

    loss_u = uncertainty_supervision_loss(
        pred=S_clean,
        target=target,
        U_sino=U_sino,
        mask=mask,
    )

    print("device:", device)
    print("input:", x.shape)
    print("S_clean:", S_clean.shape, S_clean.min().item(), S_clean.max().item())
    print("U_sino:", U_sino.shape, U_sino.min().item(), U_sino.max().item())
    print("loss:", loss.item())
    print("loss_dict:", loss_dict)
    print("loss_u:", loss_u.item())