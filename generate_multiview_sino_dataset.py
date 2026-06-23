# multiview-脚本作用：
# 用训练好的脚本，生成送给图像域的数据集
"""
CUDA_VISIBLE_DEVICES=3 python generate_multiview_sino_dataset.py \
  --config configs/sino_multiview_v12to24.yaml \
  --ckpt outputs/sino_domain_multiview_v12to24_20260606/checkpoints/sino_multiview_best_balanced.pth \
  --out-root ./multiview_dataset/sino_v12to24 \
  --splits train,valid,test \
  --views 12,15,18,21,24 \
  --batch-size 8 \
  --num-workers 0
"""

import os
import argparse
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.multiview_sino_dataset import MultiViewSinoDataset
from model.Multiview_Sino_Net import MDPR_SinoDomain
from utils.train_utils import load_config


# =========================================================
# Utils
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_sino_model_output(out):
    """
    兼容：
        return S_clean, U_sino
        return {"S_clean": ..., "U_sino": ...}
    """
    if isinstance(out, dict):
        return out["S_clean"], out["U_sino"]

    if isinstance(out, (tuple, list)) and len(out) == 2:
        return out[0], out[1]

    raise RuntimeError("模型输出格式错误，需要 return S_clean, U_sino")


def load_model_checkpoint(model, ckpt_path, device):
    print("加载 checkpoint:", ckpt_path)

    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False
    )

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print("权重格式: model_state_dict")

        if "epoch" in ckpt:
            print("checkpoint epoch:", ckpt["epoch"])
        if "best_metric" in ckpt:
            print("checkpoint best_metric:", ckpt["best_metric"])

    elif isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        print("权重格式: model")

    else:
        model.load_state_dict(ckpt)
        print("权重格式: pure state_dict")

    return model


def normalize_01(arr):
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr


def find_ct_gt_path(config, split, name):
    """
    根据 sino 文件名找到对应 ct_gt_npy。
    默认要求同名。
    """
    root = config["data"]["root"]
    ct_folder = config["data"].get("ct_folder", "ct_gt_npy")

    ct_path = os.path.join(root, split, ct_folder, name)

    if os.path.exists(ct_path):
        return ct_path

    stem = os.path.splitext(name)[0]
    ct_path = os.path.join(root, split, ct_folder, stem + ".npy")

    if os.path.exists(ct_path):
        return ct_path

    return None


def save_numpy(path, arr):
    arr = arr.astype(np.float32)
    np.save(path, arr)


# =========================================================
# Build Loader
# =========================================================

def build_loader(config, split, view, batch_size, num_workers):
    dataset = MultiViewSinoDataset(
        root=config["data"]["root"],
        split=split,
        sino_folder=config["data"].get("sino_folder", "sino_gt_npy"),
        ct_folder=config["data"].get("ct_folder", "ct_gt_npy"),
        min_views=config["data"]["min_views"],
        max_views=config["data"]["max_views"],
        test_views=view,
        train_views=None,
        use_ct=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return dataset, loader


# =========================================================
# Generate One Split / One View
# =========================================================

@torch.no_grad()
def generate_one_view(
    model,
    config,
    device,
    split,
    view,
    out_root,
    batch_size,
    num_workers,
):
    dataset, loader = build_loader(
        config=config,
        split=split,
        view=view,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    view_dir = os.path.join(out_root, split, f"view{view}")

    dirs = {
        "input_interp": os.path.join(view_dir, "input_interp_npy"),
        "pred_raw": os.path.join(view_dir, "pred_raw_npy"),
        "pred_refill": os.path.join(view_dir, "pred_refill_npy"),
        "sino_gt": os.path.join(view_dir, "sino_gt_npy"),
        "mask": os.path.join(view_dir, "mask_npy"),
        "u_sino": os.path.join(view_dir, "u_sino_npy"),
        "ct_gt": os.path.join(view_dir, "ct_gt_npy"),
    }

    for d in dirs.values():
        ensure_dir(d)

    model.eval()

    count = 0

    for x, sino_gt, mask, meta in tqdm(
        loader,
        desc=f"Generate {split} view{view}",
        leave=False
    ):
        x = x.to(device, non_blocking=True)
        sino_gt = sino_gt.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        input_interp = x[:, 0:1]

        out = model(x)
        S_clean, U_sino = parse_sino_model_output(out)

        pred_raw = torch.clamp(S_clean, 0.0, 1.0)

        # 真实推理回填：用 input_interp 的已知角度回填
        pred_refill = pred_raw * (1.0 - mask) + input_interp * mask
        pred_refill = torch.clamp(pred_refill, 0.0, 1.0)

        input_np = input_interp.detach().cpu().numpy()
        raw_np = pred_raw.detach().cpu().numpy()
        refill_np = pred_refill.detach().cpu().numpy()
        gt_np = sino_gt.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()
        u_np = U_sino.detach().cpu().numpy()

        names = meta["name"]

        bs = input_np.shape[0]

        for i in range(bs):
            name = names[i]
            stem = os.path.splitext(name)[0]
            save_name = stem + ".npy"

            save_numpy(
                os.path.join(dirs["input_interp"], save_name),
                input_np[i, 0]
            )

            save_numpy(
                os.path.join(dirs["pred_raw"], save_name),
                raw_np[i, 0]
            )

            save_numpy(
                os.path.join(dirs["pred_refill"], save_name),
                refill_np[i, 0]
            )

            save_numpy(
                os.path.join(dirs["sino_gt"], save_name),
                gt_np[i, 0]
            )

            save_numpy(
                os.path.join(dirs["mask"], save_name),
                mask_np[i, 0]
            )

            save_numpy(
                os.path.join(dirs["u_sino"], save_name),
                u_np[i, 0]
            )

            ct_path = find_ct_gt_path(
                config=config,
                split=split,
                name=name
            )

            if ct_path is not None:
                ct = np.load(ct_path)
                ct = normalize_01(ct)
                save_numpy(
                    os.path.join(dirs["ct_gt"], save_name),
                    ct
                )
            else:
                print(f"[Warning] 找不到 CT GT: split={split}, name={name}")

            count += 1

    print(f"[完成] {split} view{view}: 保存 {count} 个样本到 {view_dir}")


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/sino_multiview_v12to24.yaml",
        help="配置文件"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="正弦域多角度模型 checkpoint"
    )

    parser.add_argument(
        "--out-root",
        type=str,
        default="./multiview_dataset/sino_v12to24",
        help="输出数据集根目录"
    )

    parser.add_argument(
        "--splits",
        type=str,
        default="train,valid,test",
        help="要生成哪些 split，例如 train,valid,test"
    )

    parser.add_argument(
        "--views",
        type=str,
        default=None,
        help="要生成哪些 view，例如 12,15,18,21,24；默认用 config['data']['val_views']"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0
    )

    args = parser.parse_args()

    config = load_config(args.config)

    if args.views is None:
        views = config["data"]["val_views"]
    else:
        views = [int(v) for v in args.views.split(",")]

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    print("输出根目录:", args.out_root)
    print("splits:", splits)
    print("views:", views)

    ensure_dir(args.out_root)

    model = MDPR_SinoDomain(
        in_channels=config["model"].get("in_channels", 3),
        width=config["model"].get("width", 32),
        num_blocks=config["model"].get("num_blocks", 6),
        keep_known=config["model"].get("keep_known", False),
        norm_groups=config["model"].get("norm_groups", 8),
    ).to(device)

    model = load_model_checkpoint(
        model=model,
        ckpt_path=args.ckpt,
        device=device,
    )

    for split in splits:
        for view in views:
            generate_one_view(
                model=model,
                config=config,
                device=device,
                split=split,
                view=view,
                out_root=args.out_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

    print("\n全部生成完成 ✅")
    print("multiview dataset:", args.out_root)


if __name__ == "__main__":
    main()