import os
import argparse

import torch
import torch.optim as optim

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from datasets.multiview_sino_dataset import MultiViewSinoDataset
from model.Multiview_Sino_Net import (
    MDPR_SinoDomain,
    uncertainty_supervision_loss,
)

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
# Helper
# =========================================================

def parse_sino_model_output(out):
    if isinstance(out, dict):
        return out["S_clean"], out["U_sino"]

    if isinstance(out, (tuple, list)) and len(out) == 2:
        return out[0], out[1]

    raise RuntimeError("模型输出格式错误，需要 return S_clean, U_sino")


def get_target_views_tensor(meta, device):
    target_views = meta["target_views"]

    if isinstance(target_views, torch.Tensor):
        target_views = target_views.to(device).float()
    elif isinstance(target_views, (list, tuple)):
        target_views = torch.tensor(target_views, device=device).float()
    else:
        target_views = torch.tensor([target_views], device=device).float()

    return target_views


def weighted_sino_completion_loss(
    pred,
    target,
    mask,
    target_views=None,
    full_views=36,
    full_weight=1.0,
    unknown_weight=2.0,
    known_weight=1.0,
    view_loss_weight=True,
    view_loss_power=0.5,
    eps=1e-6,
):
    """
    多角度正弦补全 loss。

    低 view 样本权重:
        view_weight = (full_views / target_views) ** view_loss_power

    batch 内归一化，防止 loss 数值突然变大。
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    unknown = 1.0 - mask
    abs_err = torch.abs(pred - target)

    loss_full_per = abs_err.mean(dim=(1, 2, 3))

    unknown_sum = unknown.sum(dim=(1, 2, 3)) + eps
    loss_unknown_per = (abs_err * unknown).sum(dim=(1, 2, 3)) / unknown_sum

    known_sum = mask.sum(dim=(1, 2, 3)) + eps
    loss_known_per = (abs_err * mask).sum(dim=(1, 2, 3)) / known_sum

    loss_per = (
        full_weight * loss_full_per
        + unknown_weight * loss_unknown_per
        + known_weight * loss_known_per
    )

    if view_loss_weight and target_views is not None:
        view_weight = (float(full_views) / target_views).pow(view_loss_power)
        view_weight = view_weight / (view_weight.mean().detach() + eps)
        loss = (loss_per * view_weight).mean()
    else:
        view_weight = torch.ones_like(loss_per)
        loss = loss_per.mean()

    loss_dict = {
        "loss_full": loss_full_per.mean().item(),
        "loss_unknown": loss_unknown_per.mean().item(),
        "loss_known": loss_known_per.mean().item(),
        "loss_total": loss.item(),
        "view_weight_mean": view_weight.mean().item(),
    }

    return loss, loss_dict


def plot_sino_multiview_comparison(
    sparse_sino,
    sino_gt,
    pred_raw,
    pred_refill,
    U_sino,
    save_path,
):
    inp_np = sparse_sino[0, 0].detach().cpu().numpy()
    gt_np = sino_gt[0, 0].detach().cpu().numpy()
    raw_np = pred_raw[0, 0].detach().cpu().numpy()
    refill_np = pred_refill[0, 0].detach().cpu().numpy()
    u_np = U_sino[0, 0].detach().cpu().numpy()

    inp_np = inp_np.clip(0.0, 1.0)
    gt_np = gt_np.clip(0.0, 1.0)
    raw_np = raw_np.clip(0.0, 1.0)
    refill_np = refill_np.clip(0.0, 1.0)
    u_np = u_np.clip(0.0, 1.0)

    err_np = abs(refill_np - gt_np)

    images = [inp_np, raw_np, refill_np, gt_np, u_np, err_np]
    titles = ["Input", "Raw Pred", "Refill Pred", "GT", "U_sino", "Abs Error"]

    plt.figure(figsize=(26, 5))

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 6, i + 1)
        plt.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="auto")
        plt.title(title, fontsize=12)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()


def plot_multiview_metrics_curve(
    train_losses,
    val_losses,
    view_metric_history,
    val_views,
    save_path,
):
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(24, 12))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, label="Valid Avg Loss", linewidth=2)
    plt.title("Loss Curve", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 2)
    for v in val_views:
        plt.plot(
            epochs,
            view_metric_history[v]["refill_psnr"],
            label=f"View{v}",
            linewidth=1.8,
        )
    plt.plot(
        epochs,
        view_metric_history["avg"]["refill_psnr"],
        label="Avg",
        linewidth=3,
        linestyle="--",
    )
    plt.title("Refill PSNR of Each View", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("PSNR / dB")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 3)
    for v in val_views:
        plt.plot(
            epochs,
            view_metric_history[v]["refill_ssim"],
            label=f"View{v}",
            linewidth=1.8,
        )
    plt.plot(
        epochs,
        view_metric_history["avg"]["refill_ssim"],
        label="Avg",
        linewidth=3,
        linestyle="--",
    )
    plt.title("Refill SSIM of Each View", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 4)
    for v in val_views:
        plt.plot(
            epochs,
            view_metric_history[v]["raw_psnr"],
            label=f"View{v}",
            linewidth=1.8,
        )
    plt.plot(
        epochs,
        view_metric_history["avg"]["raw_psnr"],
        label="Avg",
        linewidth=3,
        linestyle="--",
    )
    plt.title("Raw PSNR of Each View", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("PSNR / dB")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# =========================================================
# Dataset / Loader
# =========================================================

def build_train_loader(config):
    batch_size = config["train"]["batch_size"]
    num_workers = config["train"]["num_workers"]
    pin_memory = config["train"].get("pin_memory", True)

    train_dataset = MultiViewSinoDataset(
        root=config["data"]["root"],
        split="train",
        sino_folder=config["data"].get("sino_folder", "sino_gt_npy"),
        ct_folder=config["data"].get("ct_folder", "ct_gt_npy"),
        min_views=config["data"]["min_views"],
        max_views=config["data"]["max_views"],
        test_views=None,
        train_views=config["data"].get("train_views", None),
        use_ct=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print("train 数据数量:", len(train_dataset))

    return train_loader


def build_fixed_view_loader(config, split, test_view):
    batch_size = config["train"]["batch_size"]
    num_workers = config["train"]["num_workers"]
    pin_memory = config["train"].get("pin_memory", True)

    dataset = MultiViewSinoDataset(
        root=config["data"]["root"],
        split=split,
        sino_folder=config["data"].get("sino_folder", "sino_gt_npy"),
        ct_folder=config["data"].get("ct_folder", "ct_gt_npy"),
        min_views=config["data"]["min_views"],
        max_views=config["data"]["max_views"],
        test_views=test_view,
        train_views=None,
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

    return loader


# =========================================================
# Train / Valid
# =========================================================

def train_one_epoch(model, train_loader, optimizer, scaler, device, config):
    model.train()

    use_amp = config["train"].get("amp", True)
    grad_clip = config["train"].get("grad_clip", 0.0)

    full_weight = config["loss"].get("full_weight", 1.0)
    unknown_weight = config["loss"].get("unknown_weight", 2.0)
    known_weight = config["loss"].get("known_weight", 1.0)
    lambda_uncert = config["loss"].get("lambda_uncert", 0.0)

    view_loss_weight = config["loss"].get("view_loss_weight", True)
    view_loss_power = config["loss"].get("view_loss_power", 0.5)

    total_loss = 0.0
    total_sino_loss = 0.0
    total_uncert_loss = 0.0

    for x, sino_gt, mask, meta in tqdm(train_loader, desc="Train", leave=False):
        x = x.to(device, non_blocking=True)
        sino_gt = sino_gt.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        target_views = get_target_views_tensor(meta, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp and torch.cuda.is_available()):
            out = model(x)
            S_clean, U_sino = parse_sino_model_output(out)

            loss_sino, loss_dict = weighted_sino_completion_loss(
                pred=S_clean,
                target=sino_gt,
                mask=mask,
                target_views=target_views,
                full_views=36,
                full_weight=full_weight,
                unknown_weight=unknown_weight,
                known_weight=known_weight,
                view_loss_weight=view_loss_weight,
                view_loss_power=view_loss_power,
            )

            if lambda_uncert > 0:
                loss_uncert = uncertainty_supervision_loss(
                    pred=S_clean,
                    target=sino_gt,
                    U_sino=U_sino,
                    mask=mask,
                    detach_error=True,
                )
            else:
                loss_uncert = torch.zeros([], device=device)

            loss = loss_sino + lambda_uncert * loss_uncert

        scaler.scale(loss).backward()

        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_sino_loss += loss_sino.item()
        total_uncert_loss += loss_uncert.item()

    n = len(train_loader)

    return {
        "loss": total_loss / n,
        "sino_loss": total_sino_loss / n,
        "uncert_loss": total_uncert_loss / n,
    }


@torch.no_grad()
def validate_one_view(model, loader, device, config, test_view, save_compare_path=None):
    model.eval()

    use_amp = config["train"].get("amp", True)

    full_weight = config["loss"].get("full_weight", 1.0)
    unknown_weight = config["loss"].get("unknown_weight", 2.0)
    known_weight = config["loss"].get("known_weight", 1.0)

    valid_loss = 0.0

    input_psnr_sum = 0.0
    input_ssim_sum = 0.0

    raw_psnr_sum = 0.0
    raw_ssim_sum = 0.0

    refill_psnr_sum = 0.0
    refill_ssim_sum = 0.0

    img_count = 0

    last_sparse = None
    last_gt = None
    last_raw = None
    last_refill = None
    last_u = None

    for x, sino_gt, mask, meta in tqdm(loader, desc=f"Valid view{test_view}", leave=False):
        x = x.to(device, non_blocking=True)
        sino_gt = sino_gt.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        sparse_sino = x[:, 0:1]

        with torch.cuda.amp.autocast(enabled=use_amp and torch.cuda.is_available()):
            out = model(x)
            S_clean, U_sino = parse_sino_model_output(out)

            target_views_tensor = torch.ones(x.size(0), device=device).float() * float(test_view)

            loss_sino, _ = weighted_sino_completion_loss(
                pred=S_clean,
                target=sino_gt,
                mask=mask,
                target_views=target_views_tensor,
                full_views=36,
                full_weight=full_weight,
                unknown_weight=unknown_weight,
                known_weight=known_weight,
                view_loss_weight=False,
                view_loss_power=0.5,
            )

        pred_raw = torch.clamp(S_clean, 0.0, 1.0)

        pred_refill = pred_raw * (1.0 - mask) + sparse_sino * mask
        pred_refill = torch.clamp(pred_refill, 0.0, 1.0)

        sparse_eval = torch.clamp(sparse_sino, 0.0, 1.0)

        valid_loss += loss_sino.item()

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

        last_sparse = sparse_eval
        last_gt = sino_gt
        last_raw = pred_raw
        last_refill = pred_refill
        last_u = U_sino

    n = len(loader)

    result = {
        "view": test_view,
        "loss": valid_loss / n,
        "input_psnr": input_psnr_sum / img_count,
        "input_ssim": input_ssim_sum / img_count,
        "raw_psnr": raw_psnr_sum / img_count,
        "raw_ssim": raw_ssim_sum / img_count,
        "refill_psnr": refill_psnr_sum / img_count,
        "refill_ssim": refill_ssim_sum / img_count,
    }

    if save_compare_path is not None and last_sparse is not None:
        plot_sino_multiview_comparison(
            sparse_sino=last_sparse,
            sino_gt=last_gt,
            pred_raw=last_raw,
            pred_refill=last_refill,
            U_sino=last_u,
            save_path=save_compare_path,
        )

    return result


def validate_all_views(model, config, device, split, epoch, exp_dir):
    val_views = config["data"]["val_views"]

    results = []

    for v in val_views:
        loader = build_fixed_view_loader(config, split=split, test_view=v)

        compare_path = None
        if v == val_views[len(val_views) // 2]:
            compare_path = os.path.join(
                exp_dir,
                "compare",
                f"epoch_{epoch:03d}_view{v}.png"
            )

        res = validate_one_view(
            model=model,
            loader=loader,
            device=device,
            config=config,
            test_view=v,
            save_compare_path=compare_path,
        )

        results.append(res)

    avg_loss = sum([r["loss"] for r in results]) / len(results)
    avg_raw_psnr = sum([r["raw_psnr"] for r in results]) / len(results)
    avg_raw_ssim = sum([r["raw_ssim"] for r in results]) / len(results)
    avg_refill_psnr = sum([r["refill_psnr"] for r in results]) / len(results)
    avg_refill_ssim = sum([r["refill_ssim"] for r in results]) / len(results)

    print("\n========== Valid Multi-view Summary ==========")
    print(f"{'View':>6} | {'Input':>10} | {'Raw':>10} | {'Refill':>10} | {'Gain':>10} | {'SSIM':>10}")
    print("-" * 75)

    for r in results:
        gain = r["refill_psnr"] - r["input_psnr"]
        print(
            f"{r['view']:>6} | "
            f"{r['input_psnr']:>10.4f} | "
            f"{r['raw_psnr']:>10.4f} | "
            f"{r['refill_psnr']:>10.4f} | "
            f"{gain:>10.4f} | "
            f"{r['refill_ssim']:>10.4f}"
        )

    print("-" * 75)
    print(f"Avg Raw PSNR    : {avg_raw_psnr:.4f}")
    print(f"Avg Raw SSIM    : {avg_raw_ssim:.4f}")
    print(f"Avg Refill PSNR : {avg_refill_psnr:.4f}")
    print(f"Avg Refill SSIM : {avg_refill_ssim:.4f}")
    print("==============================================")

    return results, {
        "avg_loss": avg_loss,
        "avg_raw_psnr": avg_raw_psnr,
        "avg_raw_ssim": avg_raw_ssim,
        "avg_refill_psnr": avg_refill_psnr,
        "avg_refill_ssim": avg_refill_ssim,
    }


def update_view_metric_history(view_metric_history, valid_results, valid_avg):
    for r in valid_results:
        v = r["view"]
        view_metric_history[v]["raw_psnr"].append(r["raw_psnr"])
        view_metric_history[v]["raw_ssim"].append(r["raw_ssim"])
        view_metric_history[v]["refill_psnr"].append(r["refill_psnr"])
        view_metric_history[v]["refill_ssim"].append(r["refill_ssim"])

    view_metric_history["avg"]["raw_psnr"].append(valid_avg["avg_raw_psnr"])
    view_metric_history["avg"]["raw_ssim"].append(valid_avg["avg_raw_ssim"])
    view_metric_history["avg"]["refill_psnr"].append(valid_avg["avg_refill_psnr"])
    view_metric_history["avg"]["refill_ssim"].append(valid_avg["avg_refill_ssim"])


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

    save_used_code_files(
        file_paths=config["code_files"],
        save_dir=os.path.join(exp_dir, "code")
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    model = MDPR_SinoDomain(
        in_channels=config["model"].get("in_channels", 3),
        width=config["model"].get("width", 32),
        num_blocks=config["model"].get("num_blocks", 6),
        keep_known=config["model"].get("keep_known", False),
        norm_groups=config["model"].get("norm_groups", 8),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total_params / 1e6:.3f} M")
    print(f"Trainable params: {trainable_params / 1e6:.3f} M")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"].get("weight_decay", 1e-4),
    )

    num_epochs = config["train"]["num_epochs"]
    use_amp = config["train"].get("amp", True)

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp and torch.cuda.is_available()
    )

    train_loader = build_train_loader(config)

    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=config["scheduler"]["warmup_epochs"],
        total_epochs=num_epochs,
        min_lr=config["scheduler"]["min_lr"]
    )

    early_stopper = EarlyStopping(
        patience=config["early_stopping"]["patience"],
        mode=config["early_stopping"]["mode"],
        min_delta=config["early_stopping"]["min_delta"]
    )

    val_views = config["data"]["val_views"]

    train_losses = []
    val_losses = []

    view_metric_history = {}

    for v in val_views:
        view_metric_history[v] = {
            "raw_psnr": [],
            "raw_ssim": [],
            "refill_psnr": [],
            "refill_ssim": [],
        }

    view_metric_history["avg"] = {
        "raw_psnr": [],
        "raw_ssim": [],
        "refill_psnr": [],
        "refill_ssim": [],
    }

    best_avg_psnr = 0.0
    best_avg_ssim = 0.0
    best_view6_ssim = 0.0
    best_balanced_score = -1e9

    log_path = os.path.join(exp_dir, "logs", "train_log.csv")

    with open(log_path, "w", encoding="utf-8") as f:
        header = [
            "epoch",
            "train_loss",
            "train_sino_loss",
            "train_uncert_loss",
            "valid_loss",
            "avg_raw_psnr",
            "avg_raw_ssim",
            "avg_refill_psnr",
            "avg_refill_ssim",
            "balanced_score",
            "lr",
        ]

        for v in val_views:
            header.extend([
                f"view{v}_input_psnr",
                f"view{v}_input_ssim",
                f"view{v}_raw_psnr",
                f"view{v}_raw_ssim",
                f"view{v}_refill_psnr",
                f"view{v}_refill_ssim",
            ])

        f.write(",".join(header) + "\n")

    for epoch in range(1, num_epochs + 1):
        print(f"\n========== Epoch [{epoch}/{num_epochs}] ==========")

        train_stat = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            config=config,
        )

        valid_results, valid_avg = validate_all_views(
            model=model,
            config=config,
            device=device,
            split="valid",
            epoch=epoch,
            exp_dir=exp_dir,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_stat["loss"])
        val_losses.append(valid_avg["avg_loss"])

        update_view_metric_history(
            view_metric_history=view_metric_history,
            valid_results=valid_results,
            valid_avg=valid_avg,
        )

        result_map = {r["view"]: r for r in valid_results}
        key_view = 6 if 6 in result_map else val_views[0]

        avg_refill_psnr = valid_avg["avg_refill_psnr"]
        avg_refill_ssim = valid_avg["avg_refill_ssim"]

        key_view_refill_psnr = result_map[key_view]["refill_psnr"]
        key_view_refill_ssim = result_map[key_view]["refill_ssim"]

        balanced_score = (
            avg_refill_psnr
            + 5.0 * avg_refill_ssim
            + 2.0 * key_view_refill_ssim
        )

        print(
            f"Epoch [{epoch}/{num_epochs}] | "
            f"Train Loss: {train_stat['loss']:.6f} | "
            f"Train Sino: {train_stat['sino_loss']:.6f} | "
            f"Train U: {train_stat['uncert_loss']:.6f} | "
            f"Avg Raw PSNR: {valid_avg['avg_raw_psnr']:.4f} | "
            f"Avg Refill PSNR: {valid_avg['avg_refill_psnr']:.4f} | "
            f"Avg Refill SSIM: {valid_avg['avg_refill_ssim']:.4f} | "
            f"View{key_view} SSIM: {key_view_refill_ssim:.4f} | "
            f"Balanced: {balanced_score:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        with open(log_path, "a", encoding="utf-8") as f:
            row = [
                epoch,
                f"{train_stat['loss']:.6f}",
                f"{train_stat['sino_loss']:.6f}",
                f"{train_stat['uncert_loss']:.6f}",
                f"{valid_avg['avg_loss']:.6f}",
                f"{valid_avg['avg_raw_psnr']:.6f}",
                f"{valid_avg['avg_raw_ssim']:.6f}",
                f"{valid_avg['avg_refill_psnr']:.6f}",
                f"{valid_avg['avg_refill_ssim']:.6f}",
                f"{balanced_score:.6f}",
                f"{current_lr:.8e}",
            ]

            for v in val_views:
                r = result_map[v]
                row.extend([
                    f"{r['input_psnr']:.6f}",
                    f"{r['input_ssim']:.6f}",
                    f"{r['raw_psnr']:.6f}",
                    f"{r['raw_ssim']:.6f}",
                    f"{r['refill_psnr']:.6f}",
                    f"{r['refill_ssim']:.6f}",
                ])

            f.write(",".join(map(str, row)) + "\n")

        plot_multiview_metrics_curve(
            train_losses=train_losses,
            val_losses=val_losses,
            view_metric_history=view_metric_history,
            val_views=val_views,
            save_path=os.path.join(exp_dir, "curve", "multiview_metrics_curve.png")
        )

        # =====================================================
        # 多策略保存 best 模型
        # =====================================================

        if avg_refill_psnr > best_avg_psnr:
            best_avg_psnr = avg_refill_psnr

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_avg_psnr,
                save_path=os.path.join(exp_dir, "checkpoints", "sino_multiview_best_psnr.pth")
            )

            print(
                f" ⭐ 保存 best_psnr 模型 | "
                f"Avg Refill PSNR = {best_avg_psnr:.4f} ⭐"
            )

        if avg_refill_ssim > best_avg_ssim:
            best_avg_ssim = avg_refill_ssim

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_avg_ssim,
                save_path=os.path.join(exp_dir, "checkpoints", "sino_multiview_best_ssim.pth")
            )

            print(
                f" ⭐ 保存 best_ssim 模型 | "
                f"Avg Refill SSIM = {best_avg_ssim:.4f} ⭐"
            )

        if key_view_refill_ssim > best_view6_ssim:
            best_view6_ssim = key_view_refill_ssim

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_view6_ssim,
                save_path=os.path.join(exp_dir, "checkpoints", f"sino_multiview_best_view{key_view}_ssim.pth")
            )

            print(
                f" ⭐ 保存 best_view{key_view}_ssim 模型 | "
                f"View{key_view} Refill SSIM = {best_view6_ssim:.4f}, "
                f"View{key_view} Refill PSNR = {key_view_refill_psnr:.4f} ⭐"
            )

        if balanced_score > best_balanced_score:
            best_balanced_score = balanced_score

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_balanced_score,
                save_path=os.path.join(exp_dir, "checkpoints", "sino_multiview_best_balanced.pth")
            )

            print(
                f" ⭐ 保存 best_balanced 模型 | "
                f"Score = {best_balanced_score:.4f}, "
                f"Avg PSNR = {avg_refill_psnr:.4f}, "
                f"Avg SSIM = {avg_refill_ssim:.4f}, "
                f"View{key_view} SSIM = {key_view_refill_ssim:.4f} ⭐"
            )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_avg_psnr,
            save_path=os.path.join(exp_dir, "checkpoints", "sino_multiview_latest.pth")
        )

        if early_stopper(balanced_score):
            print(f"早停触发：连续 {early_stopper.patience} 轮 balanced_score 没有提升")
            break

    print("\n训练完成")
    print("Best Avg Refill PSNR:", best_avg_psnr)
    print("Best Avg Refill SSIM:", best_avg_ssim)
    print(f"Best View{6 if 6 in val_views else val_views[0]} Refill SSIM:", best_view6_ssim)
    print("Best Balanced Score:", best_balanced_score)
    print("输出目录:", exp_dir)


if __name__ == "__main__":
    main()