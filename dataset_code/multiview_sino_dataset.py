import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


# =========================================================
# Basic Utils
# =========================================================

def normalize_01(arr):
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr


def natural_key(path):
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(name)
    except ValueError:
        return name


# =========================================================
# Sampling
# =========================================================

def grouped_random_indices_36(
    num_views=36,
    min_views=6,
    max_views=18,
    num_groups=3,
    seed=None,
    train_views=None,
):
    """
    分组随机采样。

    train_views=None:
        从 min_views ~ max_views 均匀随机采样。

    train_views=[6,6,6,6,7,7,8,8,9,9,10,11,12,12,15,18]:
        从 train_views 中随机抽 target_views。
        重复次数越多，该 view 出现概率越高。
    """
    rng = np.random.default_rng(seed)

    if train_views is not None and len(train_views) > 0:
        target_views = int(rng.choice(train_views))
    else:
        target_views = int(rng.integers(min_views, max_views + 1))

    groups = []
    for g in range(num_groups):
        groups.append(np.arange(g, num_views, num_groups))

    base = target_views // num_groups
    remain = target_views % num_groups

    take_counts = np.ones(num_groups, dtype=np.int64) * base

    if remain > 0:
        extra_groups = rng.choice(num_groups, size=remain, replace=False)
        take_counts[extra_groups] += 1

    selected = []

    for g, count in enumerate(take_counts):
        if count <= 0:
            continue

        chosen = rng.choice(groups[g], size=int(count), replace=False)
        selected.extend(chosen.tolist())

    selected = np.array(sorted(selected), dtype=np.int64)
    view_ratio = target_views / num_views

    return selected, target_views, take_counts, view_ratio


def fixed_group_indices_36(target_views, num_views=36, num_groups=3):
    """
    验证/测试用固定分组采样。
    保证每次 valid/test 的输入完全一致，方便公平比较。
    """
    groups = [np.arange(g, num_views, num_groups) for g in range(num_groups)]

    base = target_views // num_groups
    remain = target_views % num_groups

    take_counts = np.ones(num_groups, dtype=np.int64) * base
    take_counts[:remain] += 1

    selected = []

    for g, count in enumerate(take_counts):
        if count <= 0:
            continue

        group = groups[g]
        pos = np.linspace(0, len(group) - 1, int(count))
        pos = np.round(pos).astype(np.int64)
        selected.extend(group[pos].tolist())

    selected = np.array(sorted(selected), dtype=np.int64)
    return selected


def fixed_interval_indices_36(target_views, num_views=36, start_index=0):
    """
    固定等间隔采样。
    例如 target_views=18, start_index=0 时选 0,2,4,...,34，
    对应 1-based 角度编号 1,3,5,...,35。
    """
    target_views = int(target_views)
    start_index = int(start_index)

    if target_views <= 0 or target_views > num_views:
        raise ValueError(
            f"target_views must be in [1, {num_views}], got {target_views}"
        )

    if num_views % target_views == 0:
        step = num_views // target_views
        selected = start_index + np.arange(target_views, dtype=np.int64) * step
    else:
        step = float(num_views) / float(target_views)
        selected = np.floor(
            start_index + np.arange(target_views, dtype=np.float32) * step
        ).astype(np.int64)

    selected = np.mod(selected, num_views)
    selected = np.array(sorted(np.unique(selected)), dtype=np.int64)

    if len(selected) != target_views:
        raise RuntimeError(
            f"fixed interval sampling produced {len(selected)} unique views, "
            f"expected {target_views}: {selected.tolist()}"
        )

    return selected


def interpolate_sino_by_angle(full_sino, selected_indices):
    """
    沿角度方向做线性插值，把稀疏角度补成完整 36 角度。

    full_sino: [det, views]，例如 [367,36]
    selected_indices: 已知角度索引，例如 [0,3,6,...]

    return:
        interp_sino: [det, views]
    """
    det, num_views = full_sino.shape

    selected_indices = np.array(sorted(selected_indices), dtype=np.int64)
    all_indices = np.arange(num_views, dtype=np.float32)

    interp_sino = np.zeros_like(full_sino, dtype=np.float32)

    # 对每一个 detector 行，沿 view 方向插值
    for d in range(det):
        known_x = selected_indices.astype(np.float32)
        known_y = full_sino[d, selected_indices].astype(np.float32)

        interp_sino[d, :] = np.interp(
            all_indices,
            known_x,
            known_y
        ).astype(np.float32)

    return interp_sino


def make_sparse_sino(full_sino, selected_indices):
    """
    full_sino: [367, 36]
    selected_indices: 被保留的角度列

    return:
        x:    [3, 367, 36]
              x[0] = 插值后的 sparse_sino
              x[1] = mask
              x[2] = view_ratio_map

        y:    [1, 367, 36]
        mask: [367, 36]
        view_ratio: float
    """
    det, num_views = full_sino.shape

    mask = np.zeros_like(full_sino, dtype=np.float32)
    mask[:, selected_indices] = 1.0

    # 关键修改：未知角度不用 0，而是沿角度方向线性插值
    sparse_sino = interpolate_sino_by_angle(
        full_sino=full_sino,
        selected_indices=selected_indices
    )

    # 已知角度强制回填真实值，避免插值误差污染已知角度
    sparse_sino[:, selected_indices] = full_sino[:, selected_indices]

    view_ratio = len(selected_indices) / num_views
    view_ratio_map = np.ones_like(full_sino, dtype=np.float32) * view_ratio

    x = np.stack(
        [sparse_sino, mask, view_ratio_map],
        axis=0
    ).astype(np.float32)

    y = full_sino[None, :, :].astype(np.float32)

    return x, y, mask.astype(np.float32), view_ratio


# =========================================================
# MultiView Dataset
# =========================================================

class MultiViewSinoDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        sino_folder="sino_gt_npy",
        ct_folder="ct_gt_npy",
        min_views=6,
        max_views=18,
        test_views=None,
        train_views=None,
        train_sampling="random_group",
        eval_sampling="fixed_group",
        interval_start=0,
        use_ct=False,
    ):
        self.root = root
        self.split = split
        self.sino_folder = sino_folder
        self.ct_folder = ct_folder
        self.min_views = min_views
        self.max_views = max_views
        self.test_views = test_views
        self.train_views = train_views
        self.train_sampling = train_sampling
        self.eval_sampling = eval_sampling
        self.interval_start = interval_start
        self.use_ct = use_ct

        self.sino_dir = os.path.join(root, split, sino_folder)
        self.ct_dir = os.path.join(root, split, ct_folder)

        self.sino_paths = sorted(
            glob.glob(os.path.join(self.sino_dir, "*.npy")),
            key=natural_key
        )

        if len(self.sino_paths) == 0:
            raise RuntimeError(f"No .npy files found in {self.sino_dir}")

        if self.use_ct:
            self.ct_paths = sorted(
                glob.glob(os.path.join(self.ct_dir, "*.npy")),
                key=natural_key
            )

            if len(self.ct_paths) != len(self.sino_paths):
                raise RuntimeError(
                    f"CT count {len(self.ct_paths)} != Sino count {len(self.sino_paths)}"
                )

    def __len__(self):
        return len(self.sino_paths)

    def __getitem__(self, idx):
        sino_path = self.sino_paths[idx]

        full_sino = np.load(sino_path)
        full_sino = normalize_01(full_sino)

        if full_sino.shape != (367, 36):
            raise RuntimeError(
                f"Expected sino shape [367,36], got {full_sino.shape}: {sino_path}"
            )

        if self.split == "train":
            sampling = self.train_sampling
            if sampling in ("fixed_interval", "interval"):
                target_views = (
                    int(self.train_views[0])
                    if self.train_views is not None and len(self.train_views) > 0
                    else self.max_views
                )
                selected = fixed_interval_indices_36(
                    target_views=target_views,
                    num_views=36,
                    start_index=self.interval_start,
                )
                take_counts = None
                view_ratio = target_views / 36
            elif sampling == "fixed_group":
                target_views = (
                    int(self.train_views[0])
                    if self.train_views is not None and len(self.train_views) > 0
                    else self.max_views
                )
                selected = fixed_group_indices_36(
                    target_views=target_views,
                    num_views=36,
                    num_groups=3,
                )
                take_counts = None
                view_ratio = target_views / 36
            else:
                selected, target_views, take_counts, view_ratio = grouped_random_indices_36(
                    num_views=36,
                    min_views=self.min_views,
                    max_views=self.max_views,
                    num_groups=3,
                    seed=None,
                    train_views=self.train_views,
                )
        else:
            target_views = self.test_views if self.test_views is not None else self.max_views
            sampling = self.eval_sampling
            if sampling in ("fixed_interval", "interval"):
                selected = fixed_interval_indices_36(
                    target_views=target_views,
                    num_views=36,
                    start_index=self.interval_start,
                )
            else:
                selected = fixed_group_indices_36(
                    target_views=target_views,
                    num_views=36,
                    num_groups=3,
                )
            take_counts = None
            view_ratio = target_views / 36

        x, y, mask, view_ratio = make_sparse_sino(full_sino, selected)

        x = torch.from_numpy(x).float()                    # [3, 367, 36]
        y = torch.from_numpy(y).float()                    # [1, 367, 36]
        mask = torch.from_numpy(mask[None, :, :]).float()  # [1, 367, 36]

        selected_padded = np.full(36, -1, dtype=np.int64)
        selected_padded[:len(selected)] = selected.astype(np.int64)

        meta = {
            "name": os.path.basename(sino_path),
            "target_views": int(target_views),
            "view_ratio": float(view_ratio),
            "selected_indices": torch.from_numpy(selected_padded),
        }

        if self.use_ct:
            ct_path = self.ct_paths[idx]
            ct = np.load(ct_path)
            ct = normalize_01(ct)

            if ct.shape != (256, 256):
                raise RuntimeError(
                    f"Expected CT shape [256,256], got {ct.shape}: {ct_path}"
                )

            ct = torch.from_numpy(ct[None, :, :]).float()
            return x, y, mask, ct, meta

        return x, y, mask, meta


# =========================================================
# Fixed View Dataset
# =========================================================

class FixedViewSinoDataset(MultiViewSinoDataset):
    def __init__(
        self,
        root,
        split="train",
        sino_folder="sino_gt_npy",
        ct_folder="ct_gt_npy",
        train_fixed_view=12,
        test_views=None,
        min_views=6,
        max_views=18,
        use_ct=False,
    ):
        super().__init__(
            root=root,
            split=split,
            sino_folder=sino_folder,
            ct_folder=ct_folder,
            min_views=min_views,
            max_views=max_views,
            test_views=test_views,
            train_views=None,
            train_sampling="fixed_group",
            eval_sampling="fixed_group",
            interval_start=0,
            use_ct=use_ct,
        )
        self.train_fixed_view = int(train_fixed_view)

    def __getitem__(self, idx):
        sino_path = self.sino_paths[idx]

        full_sino = np.load(sino_path)
        full_sino = normalize_01(full_sino)

        if full_sino.shape != (367, 36):
            raise RuntimeError(
                f"Expected sino shape [367,36], got {full_sino.shape}: {sino_path}"
            )

        if self.split == "train":
            target_views = self.train_fixed_view
        else:
            target_views = self.test_views if self.test_views is not None else self.train_fixed_view

        selected = fixed_group_indices_36(
            target_views=target_views,
            num_views=36,
            num_groups=3,
        )

        x, y, mask, view_ratio = make_sparse_sino(full_sino, selected)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        mask = torch.from_numpy(mask[None, :, :]).float()

        selected_padded = np.full(36, -1, dtype=np.int64)
        selected_padded[:len(selected)] = selected.astype(np.int64)

        meta = {
            "name": os.path.basename(sino_path),
            "target_views": int(target_views),
            "view_ratio": float(view_ratio),
            "selected_indices": torch.from_numpy(selected_padded),
        }

        if self.use_ct:
            ct_path = self.ct_paths[idx]
            ct = np.load(ct_path)
            ct = normalize_01(ct)

            if ct.shape != (256, 256):
                raise RuntimeError(
                    f"Expected CT shape [256,256], got {ct.shape}: {ct_path}"
                )

            ct = torch.from_numpy(ct[None, :, :]).float()
            return x, y, mask, ct, meta

        return x, y, mask, meta