import matplotlib.pyplot as plt

# 绘制指标曲线
def plot_metrics_curve(train_losses, val_losses, val_psnrs, val_ssims, save_path):
    plt.figure(figsize=(18,6))

    #一行三列，第一个子图
    plt.subplot(1,3,1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Valid")
    plt.title("L1 Loss Evolution", fontsize=14, pad=10)
    plt.legend()  # 显示图例
    plt.grid(True, alpha=0.3) # 显示网格线

    #一行三列，第二个子图
    plt.subplot(1, 3, 2)
    plt.plot(val_psnrs, label="PSNR", linewidth=2)
    plt.title("Validation PSNR", fontsize=14, pad=10)
    plt.legend()
    plt.grid(True, alpha=0.3)

    #一行三列，第三个子图
    plt.subplot(1, 3, 3)
    plt.plot(val_ssims, label="SSIM", linewidth=2)
    plt.title("Validation SSIM", fontsize=14, pad=10)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 调整子图间距
    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=200)
    plt.close()


# 绘制CT对比图
def plot_ct_comparison(inp, gt, pred, save_path):
    inp_np = inp[0, 0].detach().cpu().numpy()
    gt_np = gt[0, 0].detach().cpu().numpy()
    pred_np = pred[0, 0].detach().cpu().numpy()

    images = [inp_np, pred_np, gt_np]
    titles = ["Input", "Prediction", "Ground Truth"]

    plt.figure(figsize=(18, 6))

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 3, i + 1)
        plt.imshow(img, cmap="gray", vmin=0, vmax=1)
        plt.title(title, fontsize=14)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()


