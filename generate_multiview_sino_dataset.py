# multiview-脚本作用：
# 用训练好的脚本，生成送给图像域的数据集
"""
CUDA_VISIBLE_DEVICES=3 python generate_multiview_sino_dataset.py \
  --config configs/sino_multiview_v12to24.yaml \
  --ckpt outputs/sino_domain_multiview_v12to24_20260623/checkpoints/sino_multiview_best_main.pth \
  --out-root ./multiview_dataset/sino_v12to24 \
  --splits train,valid,test \
  --views 12,15,18,21,24 \
  --batch-size 8 \
  --num-workers 0
"""

import os
import argparse
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_code.multiview_sino_dataset import MultiViewSinoDataset
from model.Multiview_Sino_Net import MDPR_SinoDomain
from utils.train_utils import load_config


# =========================================================
# Utils
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_sino_model_output(out):
    if isinstance(out, dict):
        return out["S_clean"]

    if isinstance(out, (tuple, list)):
        return out[0]

    if isinstance(out, torch.Tensor):
        return out

    raise RuntimeError("模型输出格式错误，需要 return S_clean")


ALLOWED_UNEXPECTED_KEY_PREFIXES = (
    "uncert_head.",
)

MODEL_CONFIG_KEYS = (
    "in_channels",
    "width",
    "num_blocks",
    "keep_known",
    "norm_groups",
)


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], "model_state_dict"

    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"], "model"

    return ckpt, "pure state_dict"


def validate_loaded_keys(incompatible):
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    bad_unexpected = [
        key for key in unexpected_keys
        if not key.startswith(ALLOWED_UNEXPECTED_KEY_PREFIXES)
    ]

    if missing_keys or bad_unexpected:
        lines = ["checkpoint 权重和当前模型不匹配，已停止生成。"]

        if missing_keys:
            lines.append("缺失的当前模型权重:")
            lines.extend([f"  - {key}" for key in missing_keys[:30]])
            if len(missing_keys) > 30:
                lines.append(f"  ... 还有 {len(missing_keys) - 30} 个")

        if bad_unexpected:
            lines.append("非兼容的多余权重:")
            lines.extend([f"  - {key}" for key in bad_unexpected[:30]])
            if len(bad_unexpected) > 30:
                lines.append(f"  ... 还有 {len(bad_unexpected) - 30} 个")

        raise RuntimeError("\n".join(lines))

    allowed_unexpected = [
        key for key in unexpected_keys
        if key.startswith(ALLOWED_UNEXPECTED_KEY_PREFIXES)
    ]
    if allowed_unexpected:
        print("忽略旧 checkpoint 中已删除的不确定性分支权重:", len(allowed_unexpected))


def find_checkpoint_config(ckpt, ckpt_path, config_path):
    if isinstance(ckpt, dict):
        ckpt_config = ckpt.get("config") or ckpt.get("train_config")
        if ckpt_config is not None:
            return ckpt_config, "checkpoint"

    exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
    yaml_candidates = sorted(
        glob.glob(os.path.join(exp_dir, "*.yaml"))
        + glob.glob(os.path.join(exp_dir, "*.yml"))
    )

    if not yaml_candidates:
        return None, None

    config_basename = os.path.basename(config_path)
    matched = [p for p in yaml_candidates if os.path.basename(p) == config_basename]

    if len(matched) == 1:
        return load_config(matched[0]), matched[0]

    if len(yaml_candidates) == 1:
        return load_config(yaml_candidates[0]), yaml_candidates[0]

    print("实验目录里找到多个 yaml，无法自动判断训练配置:")
    for candidate in yaml_candidates:
        print("  -", candidate)
    return None, None


def validate_checkpoint_config(ckpt_config, current_config, source):
    if ckpt_config is None:
        print("警告: checkpoint 或实验目录中没有找到训练配置，无法自动校验 keep_known 等模型配置。")
        return

    ckpt_model = ckpt_config.get("model", {})
    current_model = current_config.get("model", {})
    mismatches = []

    for key in MODEL_CONFIG_KEYS:
        if key not in ckpt_model:
            continue

        ckpt_value = ckpt_model.get(key)
        current_value = current_model.get(key)

        if ckpt_value != current_value:
            mismatches.append((key, ckpt_value, current_value))

    if mismatches:
        lines = [
            "生成配置和 checkpoint 训练配置不一致，已停止生成。",
            f"训练配置来源: {source}",
        ]
        for key, ckpt_value, current_value in mismatches:
            lines.append(f"  - model.{key}: checkpoint={ckpt_value}, current={current_value}")
        raise RuntimeError("\n".join(lines))

    print("checkpoint 训练配置校验通过:", source)


def load_model_checkpoint(model, ckpt_path, device, current_config, config_path):
    """
    兼容 model_state_dict / model / 纯 state_dict。
    strict=False 只用于兼容旧 uncertainty 分支，其他权重不匹配会报错。
    """
    print("加载 checkpoint:", ckpt_path)

    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False
    )

    ckpt_config, config_source = find_checkpoint_config(ckpt, ckpt_path, config_path)
    validate_checkpoint_config(ckpt_config, current_config, config_source)

    state_dict, weight_format = extract_state_dict(ckpt)
    incompatible = model.load_state_dict(state_dict, strict=False)
    validate_loaded_keys(incompatible)

    print("权重格式:", weight_format)

    if isinstance(ckpt, dict):
        if "epoch" in ckpt:
            print("checkpoint epoch:", ckpt["epoch"])
        if "best_metric" in ckpt:
            print("checkpoint best_metric:", ckpt["best_metric"])

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
        train_sampling=config["data"].get("train_sampling", "random_group"),
        eval_sampling=config["data"].get("eval_sampling", "fixed_group"),
        interval_start=config["data"].get("interval_start", 0),
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
        S_clean = parse_sino_model_output(out)

        pred_raw = torch.clamp(S_clean, 0.0, 1.0)

        # 真实推理回填：用 input_interp 的已知角度回填
        pred_refill = pred_raw * (1.0 - mask) + input_interp * mask
        pred_refill = torch.clamp(pred_refill, 0.0, 1.0)

        input_np = input_interp.detach().cpu().numpy()
        raw_np = pred_raw.detach().cpu().numpy()
        refill_np = pred_refill.detach().cpu().numpy()
        gt_np = sino_gt.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()

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
        views = [int(v.strip()) for v in args.views.split(",") if v.strip()]

    if not views:
        raise ValueError("生成 views 为空，请检查 --views 或 config['data']['val_views']")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise ValueError("生成 splits 为空，请检查 --splits")

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

    print("模型配置:", config.get("model", {}))

    model = load_model_checkpoint(
        model=model,
        ckpt_path=args.ckpt,
        device=device,
        current_config=config,
        config_path=args.config,
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