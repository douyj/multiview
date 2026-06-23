import os
import random
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func
from datetime import datetime
import shutil
import json
import yaml

#随机种子
def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#动态生成目录
def create_exp_dir(root_dir="outputs",exp_name="image_domain"):
    time_str = datetime.now().strftime("%Y%m%d")

    exp_dir = os.path.join(root_dir, f"{exp_name}_{time_str}")

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "compare"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "code"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "curve"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)

    return exp_dir
    

#计算批量的PSNR和SSIM
def calc_batch_psnr_ssim(pred, gt):
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()

    total_psnr = 0.0
    total_ssim = 0.0
    bs = pred_np.shape[0]

    for b in range(bs):
        p = pred_np[b, 0]
        g = gt_np[b, 0]

        total_psnr += psnr_func(g, p, data_range=1.0)
        total_ssim += ssim_func(g, p, data_range=1.0)

    return total_psnr, total_ssim, bs



#把 PyTorch 训练里的 Tensor 变成可以画图的二维 numpy 图像
def tensor_to_numpy_img(x):
    """
    将 Tensor / ndarray 转成可用于 imshow 的 numpy 图像。

    支持:
        Tensor: [B,C,H,W] / [C,H,W] / [H,W]
        ndarray: [B,C,H,W] / [C,H,W] / [H,W]

    返回:
        numpy.ndarray: [H,W]
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()

    if not isinstance(x, np.ndarray):
        raise TypeError(f"Unsupported type: {type(x)}")

    if x.ndim == 4:
        # [B,C,H,W] -> 取第一个样本第一个通道
        x = x[0, 0]
    elif x.ndim == 3:
        # [C,H,W] -> 取第一个通道
        x = x[0]
    elif x.ndim == 2:
        pass
    else:
        raise ValueError(f"Unsupported shape: {x.shape}")

    return x


#学习率调度器
def build_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=min_lr
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    return scheduler


#保存代码快照
def save_used_code_files(file_paths, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for file_path in file_paths:
        if not os.path.isfile(file_path):
            print(f"跳过，不存在: {file_path}")
            continue

        filename = os.path.basename(file_path)
        dst_path = os.path.join(save_dir, filename)

        shutil.copy2(file_path, dst_path)

    print(f"代码快照已保存到: {save_dir}")


#保存最佳模型
def save_checkpoint(model, optimizer, epoch, save_path, scheduler=None, best_metric=None):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, save_path)


#早停机制
class EarlyStopping:
    def __init__(self, patience=20, mode="max", min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == "max":
            improved = current_score > self.best_score + self.min_delta
        else:
            improved = current_score < self.best_score - self.min_delta

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop
    

#加载配置文件函数
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

#保存配置文件函数
def save_config(config, save_dir, config_name="config.yaml"):
    save_path = os.path.join(save_dir, config_name)

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)

    print(f"配置文件已保存: {save_path}")