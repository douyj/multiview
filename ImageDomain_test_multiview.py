"""
CUDA_VISIBLE_DEVICES=3 python ImageDomain_test_multiview.py \
  --config configs/image_multiview_fista_ours_v12to24.yaml \
  --ckpt  outputs/image_multiview_fista_ours_v12to24_20260616/checkpoints/image_multiview_best_psnr.pth \
  --split test \
  --views 12,15,18,21,24
"""

import os
import csv
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.multiview_image_dataset import MultiViewImageDataset
from model.Image_Net import ImageRestorer

from utils.train_utils import (
    calc_batch_psnr_ssim,
    load_config,
)


# =========================================================
# 模型输出解析
# =========================================================

def parse_image_model_output(out):
    if isinstance(out, dict):
        if "final_pred" in out:
            final_pred = out["final_pred"]
        elif "pred" in out:
            final_pred = out["pred"]
        else:
            raise RuntimeError("dict 输出里没有 final_pred 或 pred")

        stage1_pred = out.get("stage1_pred", None)
        return final_pred, stage1_pred

    if isinstance(out, (tuple, list)):
        if len(out) == 2:
            return out[0], out[1]
        elif len(out) == 1:
            return out[0], None

    if isinstance(out, torch.Tensor):
        return out, None

    raise RuntimeError("模型输出格式不支持，请检查 ImageRestorer.forward()")


# =========================================================
# 指标
# =========================================================

def calc_batch_rmse(pred, gt):
    """
    pred, gt: [B,1,H,W]
    返回:
        rmse_sum: 当前 batch 每张图 RMSE 之和
        bs
    """
    pred = torch.clamp(pred, 0.0, 1.0)
    gt = torch.clamp(gt, 0.0, 1.0)

    mse = torch.mean((pred - gt) ** 2, dim=(1, 2, 3))
    rmse = torch.sqrt(mse + 1e-12)

    return rmse.sum().item(), pred.shape[0]


def calc_input_metrics(x, y):
    """
    x: [B,1,H,W] 或 [B,2,H,W]
    x[:,0:1] 是 fista_ours / fista_input
    """
    input_img = torch.clamp(x[:, 0:1], 0.0, 1.0)

    psnr_sum, ssim_sum, bs = calc_batch_psnr_ssim(input_img, y)
    rmse_sum, _ = calc_batch_rmse(input_img, y)

    return psnr_sum, ssim_sum, rmse_sum, bs


# =========================================================
# 可视化
# =========================================================

def save_compare_image(x, y, pred, meta, save_path):
    """
    保存:
        Input / Prediction / GT / Input Error / Pred Error
    """
    input_img = x[0, 0].detach().cpu().numpy()
    gt = y[0, 0].detach().cpu().numpy()
    pred_np = pred[0, 0].detach().cpu().numpy()

    input_img = input_img.clip(0.0, 1.0)
    gt = gt.clip(0.0, 1.0)
    pred_np = pred_np.clip(0.0, 1.0)

    err_input = abs(input_img - gt)
    err_pred = abs(pred_np - gt)

    images = [
        input_img,
        pred_np,
        gt,
        err_input,
        err_pred,
    ]

    titles = [
        "Input",
        "Prediction",
        "CT GT",
        f"Input Error\nmean={err_input.mean():.5f}",
        f"Pred Error\nmean={err_pred.mean():.5f}",
    ]

    plt.figure(figsize=(22, 5))

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 5, i + 1)
        plt.imshow(img, cmap="gray", vmin=0, vmax=1)
        plt.title(title, fontsize=11)
        plt.axis("off")

    try:
        view = meta["view"][0].item() if hasattr(meta["view"][0], "item") else meta["view"][0]
        name = meta["name"][0]
        plt.suptitle(f"{name} | view{view}", fontsize=13)
    except Exception:
        pass

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()


# =========================================================
# checkpoint 加载
# =========================================================

def load_model_checkpoint(model, ckpt_path, device):
    print("加载 checkpoint:", ckpt_path)

    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        print("权重格式: model_state_dict")

        if "epoch" in ckpt:
            print("checkpoint epoch:", ckpt["epoch"])
        if "best_metric" in ckpt:
            print("checkpoint best_metric:", ckpt["best_metric"])

    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        print("权重格式: model")

    else:
        state_dict = ckpt
        print("权重格式: pure state_dict")

    model.load_state_dict(state_dict, strict=True)
    return model


# =========================================================
# 构建单 view loader
# =========================================================

def build_view_loader(config, split, view):
    dataset = MultiViewImageDataset(
        root=config["data"]["root"],
        split=split,
        views=[view],
        input_key=config["data"].get("input_key", "fista_ours_npy"),
        target_key=config["data"].get("target_key", "ct_gt_npy"),
        use_view_ratio=config["data"].get("use_view_ratio", True),
        full_view=config["data"].get("full_view", 36),
    )

    loader = DataLoader(
        dataset,
        batch_size=config["test"].get("batch_size", 1),
        shuffle=False,
        num_workers=config["test"].get("num_workers", 0),
        pin_memory=config["test"].get("pin_memory", True),
        drop_last=False,
    )

    return dataset, loader


# =========================================================
# 测试单个 view
# =========================================================

@torch.no_grad()
def test_one_view(
    model,
    loader,
    device,
    save_dir,
    view,
    max_save_images=20,
    save_npy=True,
):
    model.eval()

    view_dir = os.path.join(save_dir, f"view{view}")
    compare_dir = os.path.join(view_dir, "compare")
    pred_dir = os.path.join(view_dir, "pred_npy")
    input_dir = os.path.join(view_dir, "input_npy")
    gt_dir = os.path.join(view_dir, "gt_npy")

    os.makedirs(compare_dir, exist_ok=True)

    if save_npy:
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

    detail_csv = os.path.join(view_dir, "details.csv")
    os.makedirs(view_dir, exist_ok=True)

    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "name",
            "view",
            "input_psnr",
            "input_ssim",
            "input_rmse",
            "pred_psnr",
            "pred_ssim",
            "pred_rmse",
            "psnr_gain",
            "ssim_gain",
            "rmse_gain",
        ])

    input_psnr_sum = 0.0
    input_ssim_sum = 0.0
    input_rmse_sum = 0.0

    pred_psnr_sum = 0.0
    pred_ssim_sum = 0.0
    pred_rmse_sum = 0.0

    img_count = 0
    save_count = 0

    for x, y, meta in tqdm(loader, desc=f"Test view{view}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)
        final_pred, stage1_pred = parse_image_model_output(out)

        pred = torch.clamp(final_pred, 0.0, 1.0)
        input_img = torch.clamp(x[:, 0:1], 0.0, 1.0)

        p_in_sum, s_in_sum, r_in_sum, bs = calc_input_metrics(x, y)
        p_pred_sum, s_pred_sum, _ = calc_batch_psnr_ssim(pred, y)
        r_pred_sum, _ = calc_batch_rmse(pred, y)

        input_psnr_sum += p_in_sum
        input_ssim_sum += s_in_sum
        input_rmse_sum += r_in_sum

        pred_psnr_sum += p_pred_sum
        pred_ssim_sum += s_pred_sum
        pred_rmse_sum += r_pred_sum

        img_count += bs

        # 单样本详情
        for b in range(bs):
            name = meta["name"][b]

            one_input = input_img[b:b + 1]
            one_pred = pred[b:b + 1]
            one_gt = y[b:b + 1]

            one_input_psnr, one_input_ssim, _ = calc_batch_psnr_ssim(one_input, one_gt)
            one_pred_psnr, one_pred_ssim, _ = calc_batch_psnr_ssim(one_pred, one_gt)

            one_input_rmse, _ = calc_batch_rmse(one_input, one_gt)
            one_pred_rmse, _ = calc_batch_rmse(one_pred, one_gt)

            with open(detail_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    name,
                    view,
                    f"{one_input_psnr:.6f}",
                    f"{one_input_ssim:.6f}",
                    f"{one_input_rmse:.6f}",
                    f"{one_pred_psnr:.6f}",
                    f"{one_pred_ssim:.6f}",
                    f"{one_pred_rmse:.6f}",
                    f"{one_pred_psnr - one_input_psnr:.6f}",
                    f"{one_pred_ssim - one_input_ssim:.6f}",
                    f"{one_input_rmse - one_pred_rmse:.6f}",
                ])

            if save_npy:
                pred_np = one_pred[0, 0].detach().cpu().numpy().astype("float32")
                input_np = one_input[0, 0].detach().cpu().numpy().astype("float32")
                gt_np = one_gt[0, 0].detach().cpu().numpy().astype("float32")

                import numpy as np
                np.save(os.path.join(pred_dir, name + ".npy"), pred_np)
                np.save(os.path.join(input_dir, name + ".npy"), input_np)
                np.save(os.path.join(gt_dir, name + ".npy"), gt_np)

            if save_count < max_save_images:
                save_compare_image(
                    x=x[b:b + 1],
                    y=y[b:b + 1],
                    pred=pred[b:b + 1],
                    meta={
                        "name": [name],
                        "view": [view],
                    },
                    save_path=os.path.join(compare_dir, f"{save_count:03d}_{name}.png"),
                )
                save_count += 1

    avg = {
        "view": view,
        "count": img_count,
        "input_psnr": input_psnr_sum / img_count,
        "input_ssim": input_ssim_sum / img_count,
        "input_rmse": input_rmse_sum / img_count,
        "pred_psnr": pred_psnr_sum / img_count,
        "pred_ssim": pred_ssim_sum / img_count,
        "pred_rmse": pred_rmse_sum / img_count,
    }

    avg["psnr_gain"] = avg["pred_psnr"] - avg["input_psnr"]
    avg["ssim_gain"] = avg["pred_ssim"] - avg["input_ssim"]
    avg["rmse_gain"] = avg["input_rmse"] - avg["pred_rmse"]

    print(
        f"[view{view}] "
        f"Input {avg['input_psnr']:.4f}/{avg['input_ssim']:.4f}/RMSE {avg['input_rmse']:.6f} | "
        f"Pred {avg['pred_psnr']:.4f}/{avg['pred_ssim']:.4f}/RMSE {avg['pred_rmse']:.6f} | "
        f"Gain {avg['psnr_gain']:.4f}/{avg['ssim_gain']:.4f}/RMSE {avg['rmse_gain']:.6f}"
    )

    return avg


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/image_multiview_fista_ours_v12to24.yaml"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"]
    )

    parser.add_argument(
        "--views",
        type=str,
        default=None,
        help="例如 12,15,18,21,24。不填则使用 config['data']['views']"
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=None
    )

    args = parser.parse_args()

    config = load_config(args.config)

    if args.views is None:
        views = [int(v) for v in config["data"]["views"]]
    else:
        views = [int(v) for v in args.views.split(",") if v.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    model = ImageRestorer(
        in_c=config["model"]["in_c"],
        out_c=config["model"].get("out_c", 1),
        stage1_width=config["model"].get("stage1_width", 64),
        stage2_width=config["model"].get("stage2_width", 64),
        num_cab=config["model"].get("num_cab", 6),
    ).to(device)

    model = load_model_checkpoint(
        model=model,
        ckpt_path=args.ckpt,
        device=device,
    )

    model.eval()

    if args.save_dir is None:
        ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(args.ckpt)),
            f"{args.split}_results_{ckpt_name}"
        )
    else:
        save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    print("config:", args.config)
    print("ckpt:", args.ckpt)
    print("split:", args.split)
    print("views:", views)
    print("save_dir:", save_dir)

    summary_csv = os.path.join(save_dir, "summary.csv")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "view",
            "count",
            "input_psnr",
            "input_ssim",
            "input_rmse",
            "pred_psnr",
            "pred_ssim",
            "pred_rmse",
            "psnr_gain",
            "ssim_gain",
            "rmse_gain",
        ])

    all_results = []

    for view in views:
        _, loader = build_view_loader(
            config=config,
            split=args.split,
            view=view,
        )

        avg = test_one_view(
            model=model,
            loader=loader,
            device=device,
            save_dir=save_dir,
            view=view,
            max_save_images=config["test"].get("max_save_images", 20),
            save_npy=config["test"].get("save_npy", True),
        )

        all_results.append(avg)

        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                avg["view"],
                avg["count"],
                f"{avg['input_psnr']:.6f}",
                f"{avg['input_ssim']:.6f}",
                f"{avg['input_rmse']:.6f}",
                f"{avg['pred_psnr']:.6f}",
                f"{avg['pred_ssim']:.6f}",
                f"{avg['pred_rmse']:.6f}",
                f"{avg['psnr_gain']:.6f}",
                f"{avg['ssim_gain']:.6f}",
                f"{avg['rmse_gain']:.6f}",
            ])

    # AVG
    total_count = sum(r["count"] for r in all_results)

    avg_input_psnr = sum(r["input_psnr"] * r["count"] for r in all_results) / total_count
    avg_input_ssim = sum(r["input_ssim"] * r["count"] for r in all_results) / total_count
    avg_input_rmse = sum(r["input_rmse"] * r["count"] for r in all_results) / total_count

    avg_pred_psnr = sum(r["pred_psnr"] * r["count"] for r in all_results) / total_count
    avg_pred_ssim = sum(r["pred_ssim"] * r["count"] for r in all_results) / total_count
    avg_pred_rmse = sum(r["pred_rmse"] * r["count"] for r in all_results) / total_count

    avg_psnr_gain = avg_pred_psnr - avg_input_psnr
    avg_ssim_gain = avg_pred_ssim - avg_input_ssim
    avg_rmse_gain = avg_input_rmse - avg_pred_rmse

    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "AVG",
            total_count,
            f"{avg_input_psnr:.6f}",
            f"{avg_input_ssim:.6f}",
            f"{avg_input_rmse:.6f}",
            f"{avg_pred_psnr:.6f}",
            f"{avg_pred_ssim:.6f}",
            f"{avg_pred_rmse:.6f}",
            f"{avg_psnr_gain:.6f}",
            f"{avg_ssim_gain:.6f}",
            f"{avg_rmse_gain:.6f}",
        ])

    print("\n========== Image Multiview Test Summary ==========")
    print(
        f"{'View':>6} | {'Input PSNR':>11} | {'Pred PSNR':>10} | {'Gain':>8} | "
        f"{'Input SSIM':>11} | {'Pred SSIM':>10} | "
        f"{'Input RMSE':>11} | {'Pred RMSE':>10}"
    )
    print("-" * 105)

    for r in all_results:
        print(
            f"{r['view']:>6} | "
            f"{r['input_psnr']:>11.4f} | "
            f"{r['pred_psnr']:>10.4f} | "
            f"{r['psnr_gain']:>8.4f} | "
            f"{r['input_ssim']:>11.4f} | "
            f"{r['pred_ssim']:>10.4f} | "
            f"{r['input_rmse']:>11.6f} | "
            f"{r['pred_rmse']:>10.6f}"
        )

    print("-" * 105)
    print(
        f"{'AVG':>6} | "
        f"{avg_input_psnr:>11.4f} | "
        f"{avg_pred_psnr:>10.4f} | "
        f"{avg_psnr_gain:>8.4f} | "
        f"{avg_input_ssim:>11.4f} | "
        f"{avg_pred_ssim:>10.4f} | "
        f"{avg_input_rmse:>11.6f} | "
        f"{avg_pred_rmse:>10.6f}"
    )

    print("\n测试完成 ✅")
    print("summary_csv:", summary_csv)
    print("save_dir:", save_dir)


if __name__ == "__main__":
    main()