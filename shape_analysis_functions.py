import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage
from skimage.measure import label
import math
import numpy as np
from skimage.measure import label
from scipy.spatial import Delaunay
import nibabel as nb
import os

def cnh_loss(probs):
    # probs: (1, 4, D, H, W)
    device = probs.device
    pred_labels = torch.argmax(probs, dim=1)  # (1, D, H, W)

    # 二值化掩码（忽略背景）
    binary_mask = (pred_labels != 0).float()  # tensor, 可回传

    # 注意：连通域分析仍需用 numpy 来辅助，结果不回传
    mask_np = binary_mask[0].cpu().numpy().astype(np.int32)
    labeled_mask_np, num_regions = label(mask_np, connectivity=3, return_num=True)

    if num_regions == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    all_regions = []
    region_sizes = []

    # 收集每个 region 的平均概率、大小（用 numpy 辅助分区，但概率操作用 tensor 保持可导）
    for region_id in range(1, num_regions + 1):
        coords = np.where(labeled_mask_np == region_id)
        coords_tensor = tuple(torch.tensor(c, device=device, dtype=torch.long) for c in coords)

        labels_in_region = pred_labels[0][coords_tensor]  # (N,)
        probs_in_region = probs[0, labels_in_region, coords_tensor[0], coords_tensor[1], coords_tensor[2]]  # (N,)

        avg_prob = torch.mean(probs_in_region)  # 可回传
        size = probs_in_region.numel()
        region_sizes.append(size)
        all_regions.append({'coords': coords_tensor, 'avg_prob': avg_prob, 'size': size})

    # 计算 alpha
    total_size = sum(region_sizes)
    largest_size = max(region_sizes)
    largest_ratio = largest_size / (total_size + 1e-6)
    alpha = 0.01 + 0.06 * largest_ratio

    # 计算可信度
    #print(len(all_regions))
    for r in all_regions:
        #print(r['size'])
        r['credibility'] = r['avg_prob'] * (r['size'] ** alpha)

    # 找到中心区域
    center_region = max(all_regions, key=lambda x: x['credibility'])

    # 计算损失
    total_size_tensor = torch.tensor(total_size, dtype=torch.float32, device=device)
    total_credibility_tensor = torch.tensor(0.0, device=device)

    for r in all_regions:
        if r is not center_region:
            total_credibility_tensor += r['credibility']

    center_size_tensor = torch.tensor(center_region['size'], dtype=torch.float32, device=device)
    loss_tu = (1 - center_size_tensor / total_size_tensor) * total_credibility_tensor
    return loss_tu

def ih_loss(
    logits: torch.Tensor,
    window_size: int = 3,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    assert logits.dim() == 5, "logits must be (B, C=4, D, H, W)"
    B, C, D, H, W = logits.shape
    assert C == 4, "C must be 4 with channel 0 as background"
    assert window_size % 2 == 1, "window_size must be an odd number"
    
    # 1) softmax 概率
    probs = F.softmax(logits, dim=1)
    bg = probs[:, 0:1]              # (B,1,D,H,W)
    fg = probs[:, 1:]               # (B,3,D,H,W)
    fg_max, _ = fg.max(dim=1, keepdim=True)  # (B,1,D,H,W)
    
    # 2) 只惩罚背景占优（ReLU 截断）
    diff_pos = F.relu(bg - fg_max)  # (B,1,D,H,W)
    
    # 3) 邻域内累积：用 3D 卷积（核=全1），等价于在局部窗口求和
    k = window_size
    pad = k // 2
    kernel = torch.ones((1, 1, k, k, k), device=logits.device, dtype=logits.dtype)
    # groups=1，stride=1，same padding
    neighborhood_sum = F.conv3d(diff_pos, kernel, bias=None, stride=1, padding=pad)
    # (B,1,D,H,W)
    
    # 4) 全图聚合为标量损失
    if reduction == "mean":
        loss = neighborhood_sum.mean()
    elif reduction == "sum":
        loss = neighborhood_sum.sum()
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")
    
    return loss