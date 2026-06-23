import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.fbp import CTSlice_Provider

class LFISTNetLayer(nn.Module):
    def __init__(self, L_init=2e-4, lambda_reg_init=0.1):
        super(LFISTNetLayer, self).__init__()
        self.L = nn.Parameter(torch.tensor(L_init, requires_grad=True))
        self.lambda_reg = nn.Parameter(torch.tensor(lambda_reg_init, requires_grad=True))

    def forward(self, y, p_odl, iradon_func, radon_func, t, Ip):
        # 此时 p_odl 已经是 [B, 1, A, D]，直接用，效率最高
        Ay = radon_func(y)  
        grad = iradon_func(Ay - p_odl) 

        L = torch.clamp(F.softplus(self.L), 1e-6, 1e-3)
        lam = torch.clamp(F.softplus(self.lambda_reg), 1e-4, 1.0)

        d = y - L * grad
        I = torch.sign(d) * torch.clamp(torch.abs(d) - lam * L, min=0)

        tp = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        y_new = I + (t - 1) / tp * (I - Ip)

        return I, y_new, tp




class LFISTNet(nn.Module):
    # 初始化：堆叠N个迭代层
    def __init__(self, num_layers=1, L_init=2e-4, lambda_reg_init=0.1, out_range_min=0.0):
        super(LFISTNet, self).__init__()
        # 堆叠1层迭代（默认）
        self.layers = nn.ModuleList([LFISTNetLayer(L_init, lambda_reg_init) for _ in range(num_layers)])
        self.out_range_min = nn.Parameter(torch.tensor(out_range_min, dtype=torch.float32, requires_grad=True))  # 可学习的对比度参数

    # 高斯核函数（模拟MATLAB高斯滤波）    
    @staticmethod
    def gaussian_kernel(size, sigma):
        """ Function to mimic MATLAB's fspecial('gaussian', [size, size], sigma) """
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
                -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
            (size, size)
        )
        return kernel / np.sum(kernel)

    # 应用高斯滤波（给正弦图去噪）
    @staticmethod
    def apply_gaussian_filter(image, kernel):
        """ Apply Gaussian filter using conv2d """
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        kernel = kernel.to(image.device)

        padding = kernel.shape[-1] // 2
        filtered_image = F.conv2d(image, kernel, padding=padding, groups=image.shape[1])

        #  返回去噪后的干净正弦图
        return filtered_image  # .squeeze(0)  # Remove batch dimension

    # 图像对比度调整（让CT图更清晰）
    def imadjust(self,img, in_range=(0, 1), out_range=(0.2, 1)):
        """
        Adjust image contrast by stretching or shrinking intensity levels.

        Parameters:
        img (Tensor): Input image tensor.
        in_range (tuple): Minimum and maximum intensity values of the input image.
        out_range (tuple): Minimum and maximum intensity values of the output image.

        Returns:
        Tensor: Adjusted image tensor.
        """
        # Clip the image values to be within the input range
        img = torch.clamp(img, min=in_range[0], max=in_range[1])

        # Normalize the image values to the range [0, 1]
        img = (img - in_range[0]) / (in_range[1] - in_range[0])

        # Scale and shift the image values to the output range
        # img = img * (out_range[1] - out_range[0]) + out_range[0]
        out_range_min = torch.clamp(self.out_range_min, min=0.0, max=0.1)  # Ensure valid range
        img = img * (out_range[1] - out_range_min)

        #返回调好的CT图
        return img


    # 🔥 网络核心前向传播（和你的FBP强联动）
    # p: 输入稀疏正弦图
    # iradon_func/radon_func/fbp_curr: 全部来自你的 fbp.py
    def forward(self, p, iradon_func, radon_func, fbp_curr):
            # 1. 预处理：只转置一次
            p_odl = p.transpose(-1, -2) # [B, 1, D, A] -> [B, 1, A, D]
            
            # 2. 去噪（可选，建议保留）
            p_odl = self.apply_gaussian_filter(p_odl, self.gaussian_kernel(5, 0.5))

            # 3. 初始重建
            I = fbp_curr(p_odl).float()

            Ip = I.clone()
            y = I.clone()
            t = 1.0

            # 4. 迭代优化
            for layer in self.layers:
                # 传 p_odl 进去，layer 里面就不需要再 transpose 了
                I, y, t = layer(y, p_odl, iradon_func, radon_func, t, Ip)
                Ip = I

            # 5. 后处理逻辑 (保持你的代码不变)
            I = torch.clamp(I, min=0)
            I1 = I - I.min()
            den = I1.amax(dim=(1, 2, 3), keepdim=True)
            I1 = I1 / (den + 1e-7)      # 严格归一化到 [0, 1]

            I1 = I1.transpose(2, 3) 
            # I1 = self.imadjust(I1)
            I1 = torch.flip(I1, dims=[2])

            

            return I1


# ===================== 单张 FISTA 测试入口 =====================
if __name__ == "__main__":
    import os
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from skimage.metrics import peak_signal_noise_ratio as psnr_func
    from skimage.metrics import structural_similarity as ssim_func

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
            raise ValueError(f"{path} 不是二维数组: {arr.shape}")

        return arr

    def normalize01_np(x, eps=1e-8):
        x = x.astype(np.float32)
        return (x - x.min()) / (x.max() - x.min() + eps)

    def calc_metric(pred, gt):
        pred = np.clip(pred, 0.0, 1.0)
        gt = np.clip(gt, 0.0, 1.0)

        psnr = psnr_func(gt, pred, data_range=1.0)
        ssim = ssim_func(gt, pred, data_range=1.0)

        return psnr, ssim

    def find_first_common_name(sino_dir, gt_dir):
        sino_names = set(
            os.path.splitext(f)[0]
            for f in os.listdir(sino_dir)
            if f.endswith(".npy")
        )

        gt_names = set(
            os.path.splitext(f)[0]
            for f in os.listdir(gt_dir)
            if f.endswith(".npy")
        )

        common = sorted(list(sino_names & gt_names))

        if len(common) == 0:
            raise RuntimeError(
                "没有找到同名 npy 文件。\n"
                f"sino_dir: {sino_dir}\n"
                f"gt_dir: {gt_dir}"
            )

        return common[0]

    def save_compare_fig(sino_np, pred_np, gt_np, save_path, title_info=""):
        err_np = np.abs(pred_np - gt_np)

        images = [
            sino_np,
            pred_np,
            gt_np,
            err_np,
        ]

        titles = [
            "Input Sino / S_clean_refill",
            "I_fista",
            "CT GT",
            f"Abs Error\nmax={err_np.max():.5f}, mean={err_np.mean():.5f}",
        ]

        plt.figure(figsize=(20, 5))

        for i, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(1, 4, i + 1)

            if i == 0:
                plt.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="auto")
            else:
                plt.imshow(img, cmap="gray", vmin=0, vmax=1)

            plt.title(title, fontsize=11)
            plt.axis("off")

        if title_info:
            plt.suptitle(title_info, fontsize=13)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sino_dir",
        type=str,
        default="./data/dataset_THZ_uncert/test/stage2_sclean_12_npy",
        help="S_clean_refill 正弦图 npy 目录，shape 应为 [367,36]"
    )

    parser.add_argument(
        "--gt_dir",
        type=str,
        default="./data/dataset_THZ_uncert/test/ct_gt_npy",
        help="CT GT npy 目录，shape 应为 [256,256]"
    )

    parser.add_argument(
        "--provider_dir",
        type=str,
        default="./data/dataset_THZ_uncert/test/sino_gt_npy",
        help="用于初始化 ODL 算子的正弦图目录，里面有 npy/png 即可"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="文件名，不带 .npy；不填则自动取第一张同名样本"
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=7,
        help="FISTA 迭代层数"
    )

    parser.add_argument(
        "--num_view",
        type=int,
        default=36,
        help="完整正弦图角度数"
    )

    parser.add_argument(
        "--num_detectors",
        type=int,
        default=367
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="./outputs/fista_single_test/fista_compare.png",
        help="对比图保存路径"
    )

    parser.add_argument(
        "--save_npy",
        action="store_true",
        help="是否保存 I_fista npy"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("FISTA 单张测试")
    print("sino_dir     :", args.sino_dir)
    print("gt_dir       :", args.gt_dir)
    print("provider_dir :", args.provider_dir)
    print("num_layers   :", args.num_layers)
    print("num_view     :", args.num_view)
    print("num_detectors:", args.num_detectors)
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    if args.name == "":
        name = find_first_common_name(args.sino_dir, args.gt_dir)
    else:
        name = args.name

    sino_path = os.path.join(args.sino_dir, name + ".npy")
    gt_path = os.path.join(args.gt_dir, name + ".npy")

    print("测试样本:", name)
    print("sino_path:", sino_path)
    print("gt_path  :", gt_path)

    # 1. 读取正弦图和 CT GT
    sino_np = load_npy_2d(sino_path)
    gt_np = load_npy_2d(gt_path)

    # S_clean_refill 一般已经是 [0,1]，这里保险 clip
    sino_np = np.clip(sino_np, 0.0, 1.0)
    gt_np = np.clip(gt_np, 0.0, 1.0)

    if sino_np.shape != (args.num_detectors, args.num_view):
        raise ValueError(
            f"sino shape 不对: {sino_np.shape}, "
            f"期望 ({args.num_detectors}, {args.num_view})"
        )

    if gt_np.shape != (256, 256):
        raise ValueError(f"gt shape 不对: {gt_np.shape}, 期望 (256,256)")

    # 2. 初始化 ODL 算子
    print("\n初始化 CTSlice_Provider / ODL 算子...")
    provider = CTSlice_Provider(
        data_dir=args.provider_dir,
        num_view=args.num_view,
        num_detectors=args.num_detectors
    )

    # 3. 初始化 FISTA 模型
    fista_model = LFISTNet(
        num_layers=args.num_layers,
        L_init=2e-4,
        lambda_reg_init=0.1
    ).to(device)

    fista_model.eval()

    # 4. 正弦图转 tensor: [1,1,367,36]
    sino_tensor = torch.from_numpy(sino_np).float().unsqueeze(0).unsqueeze(0).to(device)

    # 5. FISTA 重建
    with torch.no_grad():
        I_fista = fista_model(
            p=sino_tensor,
            iradon_func=provider.iradon_curr,
            radon_func=provider.radon_curr,
            fbp_curr=provider.fbp_curr
        )

    pred_np = I_fista[0, 0].detach().cpu().numpy()
    pred_np = np.clip(pred_np, 0.0, 1.0)

    # 6. 计算指标
    psnr, ssim = calc_metric(pred_np, gt_np)

    print("\n" + "=" * 80)
    print("FISTA 测试完成")
    print(f"PSNR: {psnr:.6f}")
    print(f"SSIM: {ssim:.6f}")
    print("pred min/max/mean:", pred_np.min(), pred_np.max(), pred_np.mean())
    print("gt   min/max/mean:", gt_np.min(), gt_np.max(), gt_np.mean())
    print("=" * 80)

    # 7. 保存对比图
    title_info = f"{name} | FISTA-{args.num_layers} | PSNR={psnr:.4f}, SSIM={ssim:.4f}"
    save_compare_fig(
        sino_np=sino_np,
        pred_np=pred_np,
        gt_np=gt_np,
        save_path=args.save_path,
        title_info=title_info
    )

    print("对比图已保存:", args.save_path)

    # 8. 可选保存 npy
    if args.save_npy:
        npy_path = args.save_path.replace(".png", ".npy")
        np.save(npy_path, pred_np.astype(np.float32))
        print("I_fista npy 已保存:", npy_path)



#测试：
# python -m model.FISTA \
#   --sino_dir ./data/dataset_THZ_uncert/test/stage2_sclean_12_npy \
#   --gt_dir ./data/dataset_THZ_uncert/test/ct_gt_npy \
#   --provider_dir ./data/dataset_THZ_uncert/test/sino_gt_npy \
#   --name 3_31_3 \
#   --num_layers 7 \
#   --save_path ./ \
#   --save_npy