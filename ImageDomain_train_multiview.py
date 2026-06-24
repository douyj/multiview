import os
import csv
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch_msssim import ssim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_code.multiview_image_dataset import MultiViewImageDataset
from model.Image_Net import ImageRestorer

from utils.visual_utils import plot_metrics_curve
from utils.train_utils import (
    set_seed,
    create_exp_dir,
    calc_batch_psnr_ssim,
    build_warmup_cosine_scheduler,
    save_used_code_files,
    save_checkpoint,
    EarlyStopping,
    load_config,
    save_config,
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
    pred, gt: [B,1,H,W], [0,1]
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
    y: [B,1,H,W]
    """
    input_img = torch.clamp(x[:, 0:1], 0.0, 1.0)

    psnr_sum, ssim_sum, bs = calc_batch_psnr_ssim(input_img, y)
    rmse_sum, _ = calc_batch_rmse(input_img, y)

    return psnr_sum, ssim_sum, rmse_sum, bs


# =========================================================
# 可视化
# =========================================================

def plot_multiview_image_comparison(x, y, pred, meta, save_path):
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
        "Input FISTA",
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

    if isinstance(meta, dict) and "view" in meta:
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
# DataLoader
# =========================================================

def build_loader(config, split, shuffle):
    dataset = MultiViewImageDataset(
        root=config["data"]["root"],
        split=split,
        views=config["data"]["views"],
        input_key=config["data"].get("input_key", "fista_ours_npy"),
        target_key=config["data"].get("target_key", "ct_gt_npy"),
        use_view_ratio=config["data"].get("use_view_ratio", True),
        full_view=config["data"].get("full_view", 36),
    )

    batch_size = config["train"]["batch_size"] if split == "train" else config["test"].get("batch_size", config["train"]["batch_size"])
    num_workers = config["train"].get("num_workers", 0) if split == "train" else config["test"].get("num_workers", 0)
    pin_memory = config["train"].get("pin_memory", True) if split == "train" else config["test"].get("pin_memory", True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return dataset, loader


# =========================================================
# Validate
# =========================================================

@torch.no_grad()
def evaluate(model, loader, device, criterion, lambda_ssim, lambda_stage1, use_amp):
    model.eval()

    loss_sum = 0.0

    input_psnr_sum = 0.0
    input_ssim_sum = 0.0
    input_rmse_sum = 0.0

    pred_psnr_sum = 0.0
    pred_ssim_sum = 0.0
    pred_rmse_sum = 0.0

    img_count = 0

    last_x = None
    last_y = None
    last_pred = None
    last_meta = None

    for x, y, meta in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp and torch.cuda.is_available()):
            out = model(x)
            final_pred, stage1_pred = parse_image_model_output(out)

            l1_loss = criterion(final_pred, y)
            ssim_val = ssim(final_pred, y, data_range=1.0, size_average=True)
            loss = l1_loss + lambda_ssim * (1.0 - ssim_val)

            if stage1_pred is not None:
                l1_s1 = criterion(stage1_pred, y)
                ssim_s1 = ssim(stage1_pred, y, data_range=1.0, size_average=True)
                loss_s1 = l1_s1 + lambda_ssim * (1.0 - ssim_s1)
                loss = loss + lambda_stage1 * loss_s1

        pred_for_metric = torch.clamp(final_pred, 0.0, 1.0)

        loss_sum += loss.item()

        p_in, s_in, r_in, bs = calc_input_metrics(x, y)
        input_psnr_sum += p_in
        input_ssim_sum += s_in
        input_rmse_sum += r_in

        p_pred, s_pred, _ = calc_batch_psnr_ssim(pred_for_metric, y)
        r_pred, _ = calc_batch_rmse(pred_for_metric, y)

        pred_psnr_sum += p_pred
        pred_ssim_sum += s_pred
        pred_rmse_sum += r_pred

        img_count += bs

        last_x = x
        last_y = y
        last_pred = pred_for_metric
        last_meta = meta

    metrics = {
        "loss": loss_sum / len(loader),
        "input_psnr": input_psnr_sum / img_count,
        "input_ssim": input_ssim_sum / img_count,
        "input_rmse": input_rmse_sum / img_count,
        "pred_psnr": pred_psnr_sum / img_count,
        "pred_ssim": pred_ssim_sum / img_count,
        "pred_rmse": pred_rmse_sum / img_count,
        "psnr_gain": pred_psnr_sum / img_count - input_psnr_sum / img_count,
        "ssim_gain": pred_ssim_sum / img_count - input_ssim_sum / img_count,
        "rmse_gain": input_rmse_sum / img_count - pred_rmse_sum / img_count,
        "last_x": last_x,
        "last_y": last_y,
        "last_pred": last_pred,
        "last_meta": last_meta,
    }

    return metrics


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/image_multiview_fista_v12to24.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config_name = os.path.basename(args.config)

    set_seed(config["seed"])

    exp_dir = create_exp_dir(
        root_dir=config["exp"]["root_dir"],
        exp_name=config["exp"]["exp_name"]
    )

    print("输出目录:", exp_dir)

    save_config(config, exp_dir, config_name=config_name)

    if "code_files" in config:
        save_used_code_files(
            file_paths=config["code_files"],
            save_dir=os.path.join(exp_dir, "code")
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    model = ImageRestorer(
        in_c=config["model"]["in_c"],
        out_c=config["model"].get("out_c", 1),
        stage1_width=config["model"].get("stage1_width", 64),
        stage2_width=config["model"].get("stage2_width", 64),
        num_cab=config["model"].get("num_cab", 6),
    ).to(device)

    criterion = nn.L1Loss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"].get("weight_decay", 0.0),
    )

    num_epochs = config["train"]["num_epochs"]
    use_amp = config["train"].get("amp", True)
    grad_clip = config["train"].get("grad_clip", 0.0)

    lambda_stage1 = config["loss"].get("lambda_stage1", 0.1)
    lambda_ssim = config["loss"].get("lambda_ssim", 0.01)

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp and torch.cuda.is_available()
    )

    train_dataset, train_loader = build_loader(config, split="train", shuffle=True)
    valid_dataset, valid_loader = build_loader(config, split="valid", shuffle=False)

    print("train 数据数量:", len(train_dataset))
    print("valid 数据数量:", len(valid_dataset))

    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=config["scheduler"]["warmup_epochs"],
        total_epochs=num_epochs,
        min_lr=config["scheduler"]["min_lr"],
    )

    early_stopper = EarlyStopping(
        patience=config["early_stopping"]["patience"],
        mode=config["early_stopping"]["mode"],
        min_delta=config["early_stopping"]["min_delta"],
    )

    log_path = os.path.join(exp_dir, "logs", "train_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "valid_loss",
            "input_psnr",
            "input_ssim",
            "input_rmse",
            "pred_psnr",
            "pred_ssim",
            "pred_rmse",
            "psnr_gain",
            "ssim_gain",
            "rmse_gain",
            "lr",
        ])

    train_losses = []
    val_losses = []
    val_psnrs = []
    val_ssims = []

    best_psnr = -1e9
    best_rmse = 1e9

    for epoch in range(num_epochs):
        print(f"\n========== Epoch [{epoch + 1}/{num_epochs}] ==========")

        model.train()
        train_loss_sum = 0.0

        for x, y, meta in tqdm(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp and torch.cuda.is_available()):
                out = model(x)
                final_pred, stage1_pred = parse_image_model_output(out)

                l1_loss = criterion(final_pred, y)
                ssim_val = ssim(final_pred, y, data_range=1.0, size_average=True)
                loss = l1_loss + lambda_ssim * (1.0 - ssim_val)

                if stage1_pred is not None:
                    l1_s1 = criterion(stage1_pred, y)
                    ssim_s1 = ssim(stage1_pred, y, data_range=1.0, size_average=True)
                    loss_s1 = l1_s1 + lambda_ssim * (1.0 - ssim_s1)
                    loss = loss + lambda_stage1 * loss_s1

            scaler.scale(loss).backward()

            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)

        valid_metrics = evaluate(
            model=model,
            loader=valid_loader,
            device=device,
            criterion=criterion,
            lambda_ssim=lambda_ssim,
            lambda_stage1=lambda_stage1,
            use_amp=use_amp,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Valid Loss: {valid_metrics['loss']:.6f} | "
            f"Input PSNR: {valid_metrics['input_psnr']:.4f} | "
            f"Pred PSNR: {valid_metrics['pred_psnr']:.4f} | "
            f"PSNR Gain: {valid_metrics['psnr_gain']:.4f} | "
            f"Input SSIM: {valid_metrics['input_ssim']:.4f} | "
            f"Pred SSIM: {valid_metrics['pred_ssim']:.4f} | "
            f"Input RMSE: {valid_metrics['input_rmse']:.6f} | "
            f"Pred RMSE: {valid_metrics['pred_rmse']:.6f} | "
            f"RMSE Gain: {valid_metrics['rmse_gain']:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{avg_train_loss:.6f}",
                f"{valid_metrics['loss']:.6f}",
                f"{valid_metrics['input_psnr']:.4f}",
                f"{valid_metrics['input_ssim']:.4f}",
                f"{valid_metrics['input_rmse']:.6f}",
                f"{valid_metrics['pred_psnr']:.4f}",
                f"{valid_metrics['pred_ssim']:.4f}",
                f"{valid_metrics['pred_rmse']:.6f}",
                f"{valid_metrics['psnr_gain']:.4f}",
                f"{valid_metrics['ssim_gain']:.4f}",
                f"{valid_metrics['rmse_gain']:.6f}",
                f"{current_lr:.8e}",
            ])

        train_losses.append(avg_train_loss)
        val_losses.append(valid_metrics["loss"])
        val_psnrs.append(valid_metrics["pred_psnr"])
        val_ssims.append(valid_metrics["pred_ssim"])

        plot_metrics_curve(
            train_losses=train_losses,
            val_losses=val_losses,
            val_psnrs=val_psnrs,
            val_ssims=val_ssims,
            save_path=os.path.join(exp_dir, "curve", "metrics_curve.png"),
        )

        if valid_metrics["last_x"] is not None:
            plot_multiview_image_comparison(
                x=valid_metrics["last_x"],
                y=valid_metrics["last_y"],
                pred=valid_metrics["last_pred"],
                meta=valid_metrics["last_meta"],
                save_path=os.path.join(exp_dir, "compare", f"epoch_{epoch + 1:03d}.png"),
            )

        if valid_metrics["pred_psnr"] > best_psnr:
            best_psnr = valid_metrics["pred_psnr"]

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_metric=best_psnr,
                save_path=os.path.join(exp_dir, "checkpoints", "image_multiview_best_psnr.pth"),
                config=config,
            )

            print(" ⭐ 保存 best PSNR 模型 ⭐")

        if valid_metrics["pred_rmse"] < best_rmse:
            best_rmse = valid_metrics["pred_rmse"]

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_metric=best_rmse,
                save_path=os.path.join(exp_dir, "checkpoints", "image_multiview_best_rmse.pth"),
                config=config,
            )

            print(" ⭐ 保存 best RMSE 模型 ⭐")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            best_metric=best_psnr,
            save_path=os.path.join(exp_dir, "checkpoints", "image_multiview_latest.pth"),
            config=config,
        )

        if early_stopper(valid_metrics["pred_psnr"]):
            print(f"早停触发：连续 {early_stopper.patience} 轮 PSNR 没有提升")
            break


if __name__ == "__main__":
    main()