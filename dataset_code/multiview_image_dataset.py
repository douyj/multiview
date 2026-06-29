import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset


def load_npy_2d(path):
    arr = np.load(path).astype(np.float32)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"{path} 维度异常: {arr.shape}")

    if arr.ndim != 2:
        raise ValueError(f"{path} 不是二维数组: {arr.shape}")

    return arr.astype(np.float32)


class MultiViewImageDataset(Dataset):
    """
    多角度图像域恢复 Dataset。

    目录结构:
        root/split/view12/fista_ours_npy/*.npy
        root/split/view12/ct_gt_npy/*.npy
        root/split/view15/...
        ...

    input_key:
        fista_ours_npy   # 你的方法
        fista_input_npy  # baseline 后面用
    """

    def __init__(
        self,
        root,
        split="train",
        views=(12, 15, 18, 21, 24),
        input_key="fista_ours_npy",
        target_key="ct_gt_npy",
        use_view_ratio=True,
        full_view=36,
    ):
        self.root = root
        self.split = split
        self.views = [int(v) for v in views]
        self.input_key = input_key
        self.target_key = target_key
        self.use_view_ratio = bool(use_view_ratio)
        self.full_view = int(full_view)

        self.samples = []

        for view in self.views:
            view_dir = os.path.join(root, split, f"view{view}")
            input_dir = os.path.join(view_dir, input_key)
            target_dir = os.path.join(view_dir, target_key)

            if not os.path.isdir(input_dir):
                raise FileNotFoundError(f"找不到 input_dir: {input_dir}")

            if not os.path.isdir(target_dir):
                raise FileNotFoundError(f"找不到 target_dir: {target_dir}")

            input_paths = sorted(glob(os.path.join(input_dir, "*.npy")))

            for input_path in input_paths:
                filename = os.path.basename(input_path)
                target_path = os.path.join(target_dir, filename)

                if not os.path.exists(target_path):
                    print(f"[Warning] 找不到 target: {target_path}")
                    continue

                self.samples.append({
                    "input_path": input_path,
                    "target_path": target_path,
                    "filename": filename,
                    "name": os.path.splitext(filename)[0],
                    "view": view,
                    "view_ratio": float(view) / float(full_view),
                })

        if len(self.samples) == 0:
            raise RuntimeError(
                f"没有找到样本: root={root}, split={split}, "
                f"views={views}, input_key={input_key}"
            )

        print(
            f"[MultiViewImageDataset] split={split}, "
            f"input_key={input_key}, samples={len(self.samples)}, "
            f"views={self.views}, use_view_ratio={self.use_view_ratio}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        x = load_npy_2d(item["input_path"])
        y = load_npy_2d(item["target_path"])

        if x.shape != (256, 256):
            raise ValueError(f"{item['input_path']} shape={x.shape}, 期望 (256,256)")

        if y.shape != (256, 256):
            raise ValueError(f"{item['target_path']} shape={y.shape}, 期望 (256,256)")

        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)

        x = torch.from_numpy(x[None, :, :]).float()
        y = torch.from_numpy(y[None, :, :]).float()

        if self.use_view_ratio:
            ratio_map = torch.ones_like(x) * float(item["view_ratio"])
            x = torch.cat([x, ratio_map], dim=0)

        meta = {
            "name": item["name"],
            "filename": item["filename"],
            "view": item["view"],
            "view_ratio": item["view_ratio"],
            "sample_index": idx,
        }

        return x, y, meta
