import os
import argparse
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.multiview_sino_dataset import MultiViewSinoDataset
from model.Multiview_Sino_Net import MDPR_SinoDomain

from utils.train_utils import (
    calc_batch_psnr_ssim,
    load_config,
)


# =========================================================
# 基础工具
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_sino_model_output(out):
    """
    兼容两种模型输出：
        1. return S_clean, U_sino
        2. return {"S_clean": xxx, "U_sino": xxx}
    """
    if isinstance(out, dict):
        return out["S_clean"], out["U_sino"]

    if isinstance(out, (tuple, list)) and len(out) == 2:
        return out[0], out[1]

    raise RuntimeError("模型输出格式错误，需要 return S_clean, U_sino")


def load_model_checkpoint(model, ckpt_path, device):
    """
    兼容当前 train_utils.py 的保存格式：
        model_state_dict

    也兼容之前可能用过的：
        model
        纯 state_dict
    """
    print("加载权重:", ckpt_path)

    # PyTorch 2.6 默认 weights_only=True，可能导致 checkpoint 加载失败
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


# =========================================================
# 可视化
# =========================================================

def save_sino_compare(
    sparse_sino,
    pred_raw,
    pred_refill,
    sino_gt,
    U_sino,
    save_path,
    title_prefix="",
):
    """
    保存:
        Input / Raw Pred / Refill Pred / GT / U_sino / Abs Error
    """
    sparse_sino = np.clip(sparse_sino, 0.0, 1.0)
    pred_raw = np.clip(pred_raw, 0.0, 1.0)
    pred_refill = np.clip(pred_refill, 0.0, 1.0)
    sino_gt = np.clip(sino_gt, 0.0, 1.0)
    U_sino = np.clip(U_sino, 0.0, 1.0)

    abs_error = np.abs(pred_refill - sino_gt)

    images = [
        sparse_sino,
        pred_raw,
        pred_refill,
        sino_gt,
        U_sino,
        abs_error,
    ]

    titles = [
        "Input",
        "Raw Pred",
        "Refill Pred",
        "GT",
        "U_sino",
        "Abs Error",
    ]

    plt.figure(figsize=(26, 5))

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 6, i + 1)
        plt.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="auto")
        plt.title(title, fontsize=12)
        plt.axis("off")

    if title_prefix:
        plt.suptitle(title_prefix, fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()


# =========================================================
# Dataset / Loader
# =========================================================

def build_test_loader(config, split, test_view, batch_size=None, num_workers=None):
    if batch_size is None:
        batch_size = config.get("test", {}).get(
            "batch_size",
            config["train"]["batch_size"]
        )

    if num_workers is None:
        num_workers = config.get("test", {}).get(
            "num_workers",
            config["train"]["num_workers"]
        )

    pin_memory = config.get("test", {}).get(
        "pin_memory",
        config["train"].get("pin_memory", True)
    )

    dataset = MultiViewSinoDataset(
        root=config["data"]["root"],
        split=split,
        sino_folder=config["data"].get("sino_folder", "sino_gt_npy"),
        ct_folder=config["data"].get("ct_folder", "ct_gt_npy"),
        min_views=config["data"]["min_views"],
        max_views=config["data"]["max_views"],
        test_views=test_view,
        use_ct=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return dataset, loader


# =========================================================
# Test One View
# =========================================================

@torch.no_grad()
def test_one_view(
    model,
    config,
    device,
    split,
    test_view,
    save_root,
):
    dataset, loader = build_test_loader(
        config=config,
        split=split,
        test_view=test_view,
    )

    view_dir = os.path.join(save_root, f"{split}_view{test_view}")
    compare_dir = os.path.join(view_dir, "compare")
    npy_dir = os.path.join(view_dir, "npy")

    ensure_dir(view_dir)
    ensure_dir(compare_dir)

    save_npy = config.get("test", {}).get("save_npy", False)
    max_save_images = config.get("test", {}).get("max_save_images", 10)

    if save_npy:
        ensure_dir(npy_dir)

    model.eval()

    input_psnr_sum = 0.0
    input_ssim_sum = 0.0

    raw_psnr_sum = 0.0
    raw_ssim_sum = 0.0

    refill_psnr_sum = 0.0
    refill_ssim_sum = 0.0

    img_count = 0
    saved_count = 0

    detail_csv = os.path.join(view_dir, f"{split}_view{test_view}_details.csv")

    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "name",
            "target_view",
            "input_psnr",
            "input_ssim",
            "raw_psnr",
            "raw_ssim",
            "refill_psnr",
            "refill_ssim",
        ])

    for x, sino_gt, mask, meta in tqdm(loader, desc=f"Test {split} view{test_view}", leave=False):
        x = x.to(device, non_blocking=True)
        sino_gt = sino_gt.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        sparse_sino = x[:, 0:1]

        out = model(x)
        S_clean, U_sino = parse_sino_model_output(out)

        pred_raw = torch.clamp(S_clean, 0.0, 1.0)

        # 真实推理回填：只能用 sparse_sino 的已知角度，不能用 sino_gt
        pred_refill = pred_raw * (1.0 - mask) + sparse_sino * mask
        pred_refill = torch.clamp(pred_refill, 0.0, 1.0)

        sparse_eval = torch.clamp(sparse_sino, 0.0, 1.0)

        p_sum, s_sum, bs = calc_batch_psnr_ssim(sparse_eval, sino_gt)
        input_psnr_sum += p_sum
        input_ssim_sum += s_sum

        p_sum, s_sum, _ = calc_batch_psnr_ssim(pred_raw, sino_gt)
        raw_psnr_sum += p_sum
        raw_ssim_sum += s_sum

        p_sum, s_sum, _ = calc_batch_psnr_ssim(pred_refill, sino_gt)
        refill_psnr_sum += p_sum
        refill_ssim_sum += s_sum

        img_count += bs

        # ============================
        # 单样本保存
        # ============================
        sparse_np = sparse_eval.detach().cpu().numpy()
        raw_np = pred_raw.detach().cpu().numpy()
        refill_np = pred_refill.detach().cpu().numpy()
        gt_np = sino_gt.detach().cpu().numpy()
        u_np = U_sino.detach().cpu().numpy()

        names = meta["name"]

        for i in range(bs):
            name = names[i]
            stem = os.path.splitext(name)[0]

            inp_i = sparse_np[i, 0]
            raw_i = raw_np[i, 0]
            refill_i = refill_np[i, 0]
            gt_i = gt_np[i, 0]
            u_i = u_np[i, 0]

            # 这里为了 details.csv 单样本指标，复用 skimage 会更慢；
            # 简单起见，用 batch 统计为主，details 里可先不单独精确算。
            # 如果你需要每张图精确指标，后续再加 calc_single_psnr_ssim。
            from skimage.metrics import peak_signal_noise_ratio, structural_similarity

            input_psnr = peak_signal_noise_ratio(gt_i, inp_i, data_range=1.0)
            input_ssim = structural_similarity(gt_i, inp_i, data_range=1.0)

            raw_psnr = peak_signal_noise_ratio(gt_i, raw_i, data_range=1.0)
            raw_ssim = structural_similarity(gt_i, raw_i, data_range=1.0)

            refill_psnr = peak_signal_noise_ratio(gt_i, refill_i, data_range=1.0)
            refill_ssim = structural_similarity(gt_i, refill_i, data_range=1.0)

            with open(detail_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    name,
                    test_view,
                    f"{input_psnr:.6f}",
                    f"{input_ssim:.6f}",
                    f"{raw_psnr:.6f}",
                    f"{raw_ssim:.6f}",
                    f"{refill_psnr:.6f}",
                    f"{refill_ssim:.6f}",
                ])

            if saved_count < max_save_images:
                title = (
                    f"{split} view{test_view} | {name} | "
                    f"Input {input_psnr:.2f}/{input_ssim:.3f} | "
                    f"Raw {raw_psnr:.2f}/{raw_ssim:.3f} | "
                    f"Refill {refill_psnr:.2f}/{refill_ssim:.3f}"
                )

                save_path = os.path.join(
                    compare_dir,
                    f"{saved_count:03d}_{stem}_view{test_view}.png"
                )

                save_sino_compare(
                    sparse_sino=inp_i,
                    pred_raw=raw_i,
                    pred_refill=refill_i,
                    sino_gt=gt_i,
                    U_sino=u_i,
                    save_path=save_path,
                    title_prefix=title,
                )

                saved_count += 1

            if save_npy:
                sample_dir = os.path.join(npy_dir, stem)
                ensure_dir(sample_dir)

                np.save(os.path.join(sample_dir, "input.npy"), inp_i.astype(np.float32))
                np.save(os.path.join(sample_dir, "raw_pred.npy"), raw_i.astype(np.float32))
                np.save(os.path.join(sample_dir, "refill_pred.npy"), refill_i.astype(np.float32))
                np.save(os.path.join(sample_dir, "gt.npy"), gt_i.astype(np.float32))
                np.save(os.path.join(sample_dir, "u_sino.npy"), u_i.astype(np.float32))

    result = {
        "view": test_view,
        "num_samples": img_count,

        "input_psnr": input_psnr_sum / img_count,
        "input_ssim": input_ssim_sum / img_count,

        "raw_psnr": raw_psnr_sum / img_count,
        "raw_ssim": raw_ssim_sum / img_count,

        "refill_psnr": refill_psnr_sum / img_count,
        "refill_ssim": refill_ssim_sum / img_count,
    }

    return result


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/sino_multiview_v6to18.yaml",
        help="配置文件路径"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="checkpoint 路径，例如 outputs/xxx/checkpoints/sino_multiview_best.pth"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="测试哪个 split，默认 test"
    )

    parser.add_argument(
        "--views",
        type=str,
        default=None,
        help="测试 view，例如 6,9,12,15,18。默认使用 config['data']['val_views']"
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="测试结果保存目录，默认保存在实验目录/test_results"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    if args.views is not None:
        test_views = [int(v) for v in args.views.split(",")]
    else:
        test_views = config["data"]["val_views"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

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

    if args.save_dir is not None:
        save_root = args.save_dir
    else:
        # ckpt: outputs/xxx/checkpoints/xxx.pth
        exp_dir = os.path.dirname(os.path.dirname(args.ckpt))
        ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
        save_root = os.path.join(exp_dir, f"{args.split}_results_{ckpt_name}")

    ensure_dir(save_root)

    print("测试 split:", args.split)
    print("测试 views:", test_views)
    print("结果保存目录:", save_root)

    summary_csv = os.path.join(save_root, f"{args.split}_summary.csv")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "view",
            "num_samples",
            "input_psnr",
            "input_ssim",
            "raw_psnr",
            "raw_ssim",
            "refill_psnr",
            "refill_ssim",
            "raw_gain",
            "refill_gain",
        ])

    all_results = []

    for v in test_views:
        res = test_one_view(
            model=model,
            config=config,
            device=device,
            split=args.split,
            test_view=v,
            save_root=save_root,
        )

        raw_gain = res["raw_psnr"] - res["input_psnr"]
        refill_gain = res["refill_psnr"] - res["input_psnr"]

        all_results.append(res)

        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                res["view"],
                res["num_samples"],
                f"{res['input_psnr']:.6f}",
                f"{res['input_ssim']:.6f}",
                f"{res['raw_psnr']:.6f}",
                f"{res['raw_ssim']:.6f}",
                f"{res['refill_psnr']:.6f}",
                f"{res['refill_ssim']:.6f}",
                f"{raw_gain:.6f}",
                f"{refill_gain:.6f}",
            ])

    avg_input_psnr = sum([r["input_psnr"] for r in all_results]) / len(all_results)
    avg_input_ssim = sum([r["input_ssim"] for r in all_results]) / len(all_results)

    avg_raw_psnr = sum([r["raw_psnr"] for r in all_results]) / len(all_results)
    avg_raw_ssim = sum([r["raw_ssim"] for r in all_results]) / len(all_results)

    avg_refill_psnr = sum([r["refill_psnr"] for r in all_results]) / len(all_results)
    avg_refill_ssim = sum([r["refill_ssim"] for r in all_results]) / len(all_results)

    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "AVG",
            "-",
            f"{avg_input_psnr:.6f}",
            f"{avg_input_ssim:.6f}",
            f"{avg_raw_psnr:.6f}",
            f"{avg_raw_ssim:.6f}",
            f"{avg_refill_psnr:.6f}",
            f"{avg_refill_ssim:.6f}",
            f"{avg_raw_psnr - avg_input_psnr:.6f}",
            f"{avg_refill_psnr - avg_input_psnr:.6f}",
        ])

    print("\n========== Test Summary ==========")
    print(
        f"{'View':>6} | {'Input PSNR':>11} | {'Input SSIM':>11} | "
        f"{'Raw PSNR':>10} | {'Raw SSIM':>9} | "
        f"{'Refill PSNR':>11} | {'Refill SSIM':>11}"
    )
    print("-" * 95)

    for r in all_results:
        print(
            f"{r['view']:>6} | "
            f"{r['input_psnr']:>11.4f} | "
            f"{r['input_ssim']:>11.4f} | "
            f"{r['raw_psnr']:>10.4f} | "
            f"{r['raw_ssim']:>9.4f} | "
            f"{r['refill_psnr']:>11.4f} | "
            f"{r['refill_ssim']:>11.4f}"
        )

    print("-" * 95)
    print(
        f"{'AVG':>6} | "
        f"{avg_input_psnr:>11.4f} | "
        f"{avg_input_ssim:>11.4f} | "
        f"{avg_raw_psnr:>10.4f} | "
        f"{avg_raw_ssim:>9.4f} | "
        f"{avg_refill_psnr:>11.4f} | "
        f"{avg_refill_ssim:>11.4f}"
    )
    print("==================================")
    print("summary_csv:", summary_csv)
    print("save_root:", save_root)


if __name__ == "__main__":
    main()