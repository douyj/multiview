"""
CUDA_VISIBLE_DEVICES=3 python generate_multiview_fista_dataset.py \
  --sino-root ./multiview_dataset/sino_v12to24 \
  --out-root ./multiview_dataset/fista_v12to24 \
  --splits train,valid,test \
  --views 12,15,18,21,24 \
  --full-view 36 \
  --num-detectors 367 \
  --num-layers 7 \
  --batch-size 2 \
  --max-compare 20 \
  --overwrite
"""

import os
import csv
import argparse
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

from model.fbp import CTSlice_Provider
from model.FISTA import LFISTNet


# =========================================================
# 基础工具
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def natural_key(path):
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(name)
    except ValueError:
        return name


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
        raise ValueError(f"{path} 不是二维 npy: {arr.shape}")

    return arr.astype(np.float32)


def save_npy(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr.astype(np.float32))


def normalize_for_show(x, eps=1e-8):
    x = x.astype(np.float32)
    x_min = x.min()
    x_max = x.max()

    if x_max - x_min < eps:
        return x * 0.0

    return (x - x_min) / (x_max - x_min + eps)


def calc_one_metric(pred, gt):
    pred = np.clip(pred, 0.0, 1.0)
    gt = np.clip(gt, 0.0, 1.0)

    psnr = psnr_func(gt, pred, data_range=1.0)
    ssim = ssim_func(gt, pred, data_range=1.0)

    return psnr, ssim


def load_optional_fista_ckpt(model, ckpt_path, device):
    if ckpt_path is None or ckpt_path == "":
        print("未提供 FISTA checkpoint，使用 LFISTNet 默认参数。")
        return model

    print("加载 FISTA checkpoint:", ckpt_path)

    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False
    )

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)

    if isinstance(ckpt, dict) and "epoch" in ckpt:
        print("checkpoint epoch:", ckpt["epoch"])

    if isinstance(ckpt, dict) and "best_metric" in ckpt:
        print("checkpoint best_metric:", ckpt["best_metric"])

    return model


# =========================================================
# 可视化
# =========================================================

def plot_fista_compare(
    sino_input,
    I_input,
    I_ours,
    I_gt_sino,
    ct_gt,
    save_path,
    title_info=""
):
    sino_show = normalize_for_show(sino_input)

    I_input = np.clip(I_input, 0.0, 1.0)
    I_ours = np.clip(I_ours, 0.0, 1.0)
    I_gt_sino = np.clip(I_gt_sino, 0.0, 1.0)
    ct_gt = np.clip(ct_gt, 0.0, 1.0)

    err_input = np.abs(I_input - ct_gt)
    err_ours = np.abs(I_ours - ct_gt)

    images = [
        sino_show,
        I_input,
        I_ours,
        I_gt_sino,
        ct_gt,
        err_input,
        err_ours,
    ]

    titles = [
        "Input Sino",
        "FISTA Input",
        "FISTA Ours",
        "FISTA Full Sino",
        "CT GT",
        "Err Input",
        "Err Ours",
    ]

    plt.figure(figsize=(28, 5))

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)

        if i == 0:
            plt.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="auto")
        else:
            plt.imshow(img, cmap="gray", vmin=0, vmax=1)

        plt.title(title, fontsize=10)
        plt.axis("off")

    if title_info:
        plt.suptitle(title_info, fontsize=13)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()


# =========================================================
# Dataset
# =========================================================

class MultiViewSinoForFISTADataset(Dataset):
    """
    读取：

    multiview_dataset/sino_v12to24/{split}/view{view}/
        input_interp_npy/
        pred_refill_npy/
        sino_gt_npy/
        ct_gt_npy/
    """

    def __init__(self, sino_root, split, view):
        self.sino_root = sino_root
        self.split = split
        self.view = int(view)

        self.view_dir = os.path.join(sino_root, split, f"view{view}")

        self.input_dir = os.path.join(self.view_dir, "input_interp_npy")
        self.ours_dir = os.path.join(self.view_dir, "pred_refill_npy")
        self.gt_sino_dir = os.path.join(self.view_dir, "sino_gt_npy")
        self.ct_gt_dir = os.path.join(self.view_dir, "ct_gt_npy")

        for d in [
            self.input_dir,
            self.ours_dir,
            self.gt_sino_dir,
            self.ct_gt_dir,
        ]:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"找不到目录: {d}")

        self.input_paths = sorted(
            glob(os.path.join(self.input_dir, "*.npy")),
            key=natural_key
        )

        if len(self.input_paths) == 0:
            raise RuntimeError(f"{self.input_dir} 中没有 npy 文件")

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        input_path = self.input_paths[idx]
        filename = os.path.basename(input_path)
        name = os.path.splitext(filename)[0]

        ours_path = os.path.join(self.ours_dir, filename)
        gt_sino_path = os.path.join(self.gt_sino_dir, filename)
        ct_gt_path = os.path.join(self.ct_gt_dir, filename)

        if not os.path.exists(ours_path):
            raise FileNotFoundError(f"找不到 pred_refill: {ours_path}")

        if not os.path.exists(gt_sino_path):
            raise FileNotFoundError(f"找不到 sino_gt: {gt_sino_path}")

        if not os.path.exists(ct_gt_path):
            raise FileNotFoundError(f"找不到 ct_gt: {ct_gt_path}")

        input_sino = load_npy_2d(input_path)
        ours_sino = load_npy_2d(ours_path)
        gt_sino = load_npy_2d(gt_sino_path)
        ct_gt = load_npy_2d(ct_gt_path)

        if input_sino.shape != (367, 36):
            raise ValueError(f"{input_path} shape={input_sino.shape}, 期望 (367,36)")
        if ours_sino.shape != (367, 36):
            raise ValueError(f"{ours_path} shape={ours_sino.shape}, 期望 (367,36)")
        if gt_sino.shape != (367, 36):
            raise ValueError(f"{gt_sino_path} shape={gt_sino.shape}, 期望 (367,36)")
        if ct_gt.shape != (256, 256):
            raise ValueError(f"{ct_gt_path} shape={ct_gt.shape}, 期望 (256,256)")

        input_sino = np.clip(input_sino, 0.0, 1.0)
        ours_sino = np.clip(ours_sino, 0.0, 1.0)
        gt_sino = np.clip(gt_sino, 0.0, 1.0)
        ct_gt = np.clip(ct_gt, 0.0, 1.0)

        input_sino = torch.from_numpy(input_sino[None, :, :]).float()  # [1,367,36]
        ours_sino = torch.from_numpy(ours_sino[None, :, :]).float()
        gt_sino = torch.from_numpy(gt_sino[None, :, :]).float()

        return input_sino, ours_sino, gt_sino, ct_gt.astype(np.float32), name


# =========================================================
# 处理一个 split/view
# =========================================================

@torch.no_grad()
def process_one_split_view(
    sino_root,
    out_root,
    split,
    view,
    provider,
    fista_model,
    device,
    batch_size,
    num_workers,
    max_compare,
    overwrite,
):
    print("\n" + "=" * 80)
    print(f"开始 FISTA: split={split}, view={view}")
    print("sino_root:", sino_root)
    print("out_root :", out_root)
    print("=" * 80)

    dataset = MultiViewSinoForFISTADataset(
        sino_root=sino_root,
        split=split,
        view=view
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    out_view_dir = os.path.join(out_root, split, f"view{view}")

    fista_input_dir = os.path.join(out_view_dir, "fista_input_npy")
    fista_ours_dir = os.path.join(out_view_dir, "fista_ours_npy")
    fista_gt_sino_dir = os.path.join(out_view_dir, "fista_gt_sino_npy")
    ct_gt_dir = os.path.join(out_view_dir, "ct_gt_npy")
    compare_dir = os.path.join(out_view_dir, "compare_png")

    for d in [
        fista_input_dir,
        fista_ours_dir,
        fista_gt_sino_dir,
        ct_gt_dir,
        compare_dir,
    ]:
        os.makedirs(d, exist_ok=True)

    metrics_csv = os.path.join(out_view_dir, "metrics.csv")

    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "name",
            "view",
            "fista_input_psnr",
            "fista_input_ssim",
            "fista_ours_psnr",
            "fista_ours_ssim",
            "fista_gt_sino_psnr",
            "fista_gt_sino_ssim",
            "ours_gain_psnr",
            "ours_gain_ssim",
        ])

    sum_input_psnr = 0.0
    sum_input_ssim = 0.0
    sum_ours_psnr = 0.0
    sum_ours_ssim = 0.0
    sum_gt_sino_psnr = 0.0
    sum_gt_sino_ssim = 0.0

    metric_count = 0
    compare_count = 0

    fista_model.eval()

    for input_sino, ours_sino, gt_sino, ct_gt_batch, names in tqdm(
        loader,
        desc=f"FISTA {split} view{view}",
        leave=False
    ):
        input_sino = input_sino.to(device, non_blocking=True)  # [B,1,367,36]
        ours_sino = ours_sino.to(device, non_blocking=True)
        gt_sino = gt_sino.to(device, non_blocking=True)

        input_sino = torch.clamp(input_sino, 0.0, 1.0)
        ours_sino = torch.clamp(ours_sino, 0.0, 1.0)
        gt_sino = torch.clamp(gt_sino, 0.0, 1.0)

        # 核心：直接调用你的 LFISTNet.forward()
        I_input = fista_model(
            p=input_sino,
            iradon_func=provider.iradon_curr,
            radon_func=provider.radon_curr,
            fbp_curr=provider.fbp_curr
        )

        I_ours = fista_model(
            p=ours_sino,
            iradon_func=provider.iradon_curr,
            radon_func=provider.radon_curr,
            fbp_curr=provider.fbp_curr
        )

        I_gt_sino = fista_model(
            p=gt_sino,
            iradon_func=provider.iradon_curr,
            radon_func=provider.radon_curr,
            fbp_curr=provider.fbp_curr
        )

        bs = input_sino.shape[0]

        for b in range(bs):
            name = names[b]

            input_path = os.path.join(fista_input_dir, name + ".npy")
            ours_path = os.path.join(fista_ours_dir, name + ".npy")
            gt_sino_path = os.path.join(fista_gt_sino_dir, name + ".npy")
            ct_gt_path = os.path.join(ct_gt_dir, name + ".npy")

            if (
                not overwrite
                and os.path.exists(input_path)
                and os.path.exists(ours_path)
                and os.path.exists(gt_sino_path)
                and os.path.exists(ct_gt_path)
            ):
                continue

            input_np = I_input[b, 0].detach().cpu().numpy()
            ours_np = I_ours[b, 0].detach().cpu().numpy()
            gt_sino_np = I_gt_sino[b, 0].detach().cpu().numpy()

            input_np = np.clip(input_np, 0.0, 1.0)
            ours_np = np.clip(ours_np, 0.0, 1.0)
            gt_sino_np = np.clip(gt_sino_np, 0.0, 1.0)

            ct_gt_np = ct_gt_batch[b].numpy().astype(np.float32)
            ct_gt_np = np.clip(ct_gt_np, 0.0, 1.0)

            save_npy(input_path, input_np)
            save_npy(ours_path, ours_np)
            save_npy(gt_sino_path, gt_sino_np)
            save_npy(ct_gt_path, ct_gt_np)

            input_psnr, input_ssim = calc_one_metric(input_np, ct_gt_np)
            ours_psnr, ours_ssim = calc_one_metric(ours_np, ct_gt_np)
            gt_sino_psnr, gt_sino_ssim = calc_one_metric(gt_sino_np, ct_gt_np)

            sum_input_psnr += input_psnr
            sum_input_ssim += input_ssim
            sum_ours_psnr += ours_psnr
            sum_ours_ssim += ours_ssim
            sum_gt_sino_psnr += gt_sino_psnr
            sum_gt_sino_ssim += gt_sino_ssim
            metric_count += 1

            with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    name,
                    view,
                    f"{input_psnr:.6f}",
                    f"{input_ssim:.6f}",
                    f"{ours_psnr:.6f}",
                    f"{ours_ssim:.6f}",
                    f"{gt_sino_psnr:.6f}",
                    f"{gt_sino_ssim:.6f}",
                    f"{ours_psnr - input_psnr:.6f}",
                    f"{ours_ssim - input_ssim:.6f}",
                ])

            if compare_count < max_compare:
                compare_path = os.path.join(compare_dir, name + ".png")

                title_info = (
                    f"{split} view{view} | {name} | "
                    f"Input {input_psnr:.2f}/{input_ssim:.3f} | "
                    f"Ours {ours_psnr:.2f}/{ours_ssim:.3f}"
                )

                # 这里用 ours_sino 做正弦图展示
                sino_show_np = ours_sino[b, 0].detach().cpu().numpy()

                plot_fista_compare(
                    sino_input=sino_show_np,
                    I_input=I_input[b, 0].detach().cpu().numpy(),
                    I_ours=I_ours[b, 0].detach().cpu().numpy(),
                    I_gt_sino=I_gt_sino[b, 0].detach().cpu().numpy(),
                    ct_gt=ct_gt_np,
                    save_path=compare_path,
                    title_info=title_info
                )

                compare_count += 1

    if metric_count == 0:
        raise RuntimeError(f"{split} view{view} 没有生成任何样本")

    avg = {
        "split": split,
        "view": view,
        "count": metric_count,
        "input_psnr": sum_input_psnr / metric_count,
        "input_ssim": sum_input_ssim / metric_count,
        "ours_psnr": sum_ours_psnr / metric_count,
        "ours_ssim": sum_ours_ssim / metric_count,
        "gt_sino_psnr": sum_gt_sino_psnr / metric_count,
        "gt_sino_ssim": sum_gt_sino_ssim / metric_count,
    }

    with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "AVG",
            view,
            f"{avg['input_psnr']:.6f}",
            f"{avg['input_ssim']:.6f}",
            f"{avg['ours_psnr']:.6f}",
            f"{avg['ours_ssim']:.6f}",
            f"{avg['gt_sino_psnr']:.6f}",
            f"{avg['gt_sino_ssim']:.6f}",
            f"{avg['ours_psnr'] - avg['input_psnr']:.6f}",
            f"{avg['ours_ssim'] - avg['input_ssim']:.6f}",
        ])

    print(
        f"[完成] {split} view{view} | "
        f"N={avg['count']} | "
        f"Input {avg['input_psnr']:.4f}/{avg['input_ssim']:.4f} | "
        f"Ours {avg['ours_psnr']:.4f}/{avg['ours_ssim']:.4f} | "
        f"Gain {avg['ours_psnr'] - avg['input_psnr']:.4f} | "
        f"FullSino {avg['gt_sino_psnr']:.4f}/{avg['gt_sino_ssim']:.4f}"
    )

    return avg


# =========================================================
# 主函数
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sino-root",
        type=str,
        default="./multiview_dataset/sino_v12to24",
        help="generate_multiview_sino_dataset.py 输出的数据集目录"
    )

    parser.add_argument(
        "--out-root",
        type=str,
        default="./multiview_dataset/fista_v12to24",
        help="FISTA 输出目录"
    )

    parser.add_argument(
        "--splits",
        type=str,
        default="train,valid,test"
    )

    parser.add_argument(
        "--views",
        type=str,
        default="12,15,18,21,24"
    )

    parser.add_argument(
        "--provider-dir",
        type=str,
        default="",
        help="用于初始化 CTSlice_Provider 的目录。不填则自动用 sino-root 下第一个 split/view 的 sino_gt_npy"
    )

    parser.add_argument(
        "--full-view",
        type=int,
        default=36
    )

    parser.add_argument(
        "--num-detectors",
        type=int,
        default=367
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=7
    )

    parser.add_argument(
        "--fista-ckpt",
        type=str,
        default="",
        help="可选：如果你训练过 LFISTNet，就填 checkpoint；不填则用默认参数"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="FISTA 比 FBP 慢，建议先 1 或 2"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--max-compare",
        type=int,
        default=20
    )

    parser.add_argument(
        "--overwrite",
        action="store_true"
    )

    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    views = [int(v) for v in args.views.split(",") if v.strip()]

    print("=" * 80)
    print("sino_root    :", args.sino_root)
    print("out_root     :", args.out_root)
    print("splits       :", splits)
    print("views        :", views)
    print("full_view    :", args.full_view)
    print("num_detectors:", args.num_detectors)
    print("num_layers   :", args.num_layers)
    print("fista_ckpt   :", args.fista_ckpt)
    print("batch_size   :", args.batch_size)
    print("=" * 80)

    os.makedirs(args.out_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # provider 初始化目录
    if args.provider_dir:
        provider_dir = args.provider_dir
    else:
        provider_dir = os.path.join(
            args.sino_root,
            splits[0],
            f"view{views[0]}",
            "sino_gt_npy"
        )

    if not os.path.isdir(provider_dir):
        raise FileNotFoundError(f"provider_dir 不存在: {provider_dir}")

    print("\n初始化 CTSlice_Provider / ODL 算子...")
    print("provider_dir:", provider_dir)

    provider = CTSlice_Provider(
        data_dir=provider_dir,
        num_view=args.full_view,
        num_detectors=args.num_detectors
    )

    print("\n初始化 LFISTNet...")
    fista_model = LFISTNet(
        num_layers=args.num_layers,
        L_init=2e-4,
        lambda_reg_init=0.1
    ).to(device)

    fista_model = load_optional_fista_ckpt(
        model=fista_model,
        ckpt_path=args.fista_ckpt,
        device=device
    )

    fista_model.eval()

    summary_csv = os.path.join(args.out_root, "fista_summary.csv")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split",
            "view",
            "count",
            "fista_input_psnr",
            "fista_input_ssim",
            "fista_ours_psnr",
            "fista_ours_ssim",
            "fista_gt_sino_psnr",
            "fista_gt_sino_ssim",
            "ours_gain_psnr",
            "ours_gain_ssim",
        ])

    all_results = []

    for split in splits:
        for view in views:
            avg = process_one_split_view(
                sino_root=args.sino_root,
                out_root=args.out_root,
                split=split,
                view=view,
                provider=provider,
                fista_model=fista_model,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_compare=args.max_compare,
                overwrite=args.overwrite,
            )

            all_results.append(avg)

            with open(summary_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    avg["split"],
                    avg["view"],
                    avg["count"],
                    f"{avg['input_psnr']:.6f}",
                    f"{avg['input_ssim']:.6f}",
                    f"{avg['ours_psnr']:.6f}",
                    f"{avg['ours_ssim']:.6f}",
                    f"{avg['gt_sino_psnr']:.6f}",
                    f"{avg['gt_sino_ssim']:.6f}",
                    f"{avg['ours_psnr'] - avg['input_psnr']:.6f}",
                    f"{avg['ours_ssim'] - avg['input_ssim']:.6f}",
                ])

    print("\n========== FISTA Summary ==========")

    for split in splits:
        split_results = [r for r in all_results if r["split"] == split]

        avg_input_psnr = sum(r["input_psnr"] for r in split_results) / len(split_results)
        avg_input_ssim = sum(r["input_ssim"] for r in split_results) / len(split_results)

        avg_ours_psnr = sum(r["ours_psnr"] for r in split_results) / len(split_results)
        avg_ours_ssim = sum(r["ours_ssim"] for r in split_results) / len(split_results)

        avg_gt_sino_psnr = sum(r["gt_sino_psnr"] for r in split_results) / len(split_results)

        print(f"\n[{split}]")
        print(
            f"{'View':>6} | {'Input':>10} | {'Ours':>10} | "
            f"{'Gain':>10} | {'Ours SSIM':>10} | {'FullSino':>10}"
        )
        print("-" * 75)

        for r in split_results:
            print(
                f"{r['view']:>6} | "
                f"{r['input_psnr']:>10.4f} | "
                f"{r['ours_psnr']:>10.4f} | "
                f"{r['ours_psnr'] - r['input_psnr']:>10.4f} | "
                f"{r['ours_ssim']:>10.4f} | "
                f"{r['gt_sino_psnr']:>10.4f}"
            )

        print("-" * 75)
        print(
            f"{'AVG':>6} | "
            f"{avg_input_psnr:>10.4f} | "
            f"{avg_ours_psnr:>10.4f} | "
            f"{avg_ours_psnr - avg_input_psnr:>10.4f} | "
            f"{avg_ours_ssim:>10.4f} | "
            f"{avg_gt_sino_psnr:>10.4f}"
        )

    print("\n全部 multiview FISTA 数据生成完成 ✅")
    print("输出目录:", args.out_root)
    print("summary_csv:", summary_csv)


if __name__ == "__main__":
    main()