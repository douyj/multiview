import os
import odl
import torch
import numpy as np
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from odl.contrib import torch as odl_torch


# 用途：提供CT扫描数据集
# 输入：批量图像（支持 png / npy）
# 输出：包含 Radon 变换、IRadon 变换、FBP 重建结果、CT 算子的元组
class CTSlice_Provider(Dataset):

    def __init__(self, data_dir, poission_level=5e6, gaussian_level=0.05,
                 num_view=36, num_detectors=367):
        self.data_dir = data_dir

        png_list = sorted(glob(os.path.join(data_dir, "*.png")))
        npy_list = sorted(glob(os.path.join(data_dir, "*.npy")))

        # 同时支持 png 和 npy
        self.file_list = sorted(png_list + npy_list)

        if len(self.file_list) == 0:
            raise FileNotFoundError(f"在 {data_dir} 中没有找到 png 或 npy 文件")

        self.poission_level = poission_level
        self.gaussian_level = gaussian_level
        self.num_view = num_view
        self.num_detectors = num_detectors

        # 1. 全视图CT算子（180个角度）
        self.radon_full, self.iradon_full, self.fbp_full, self.op_norm_full = \
            self._radon_transform(num_view=180, num_detectors=num_detectors)

        # 2. 当前设置视图数量的CT算子
        self.radon_curr, self.iradon_curr, self.fbp_curr, self.op_norm_curr = \
            self._radon_transform(num_view=num_view, num_detectors=num_detectors)

    def __len__(self):
        return len(self.file_list)

    # 造CT算子的函数
    def _radon_transform(self, num_view=36, num_detectors=367):
        xx = 200

        space = odl.uniform_discr(
            [-xx, -xx], [xx, xx], [256, 256], dtype='float32'
        )

        geometry = odl.tomo.parallel_beam_geometry(
            space, num_angles=num_view, det_shape=num_detectors
        )

        operator = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

        op_norm = odl.operator.power_method_opnorm(operator)
        op_norm = torch.from_numpy(np.array(op_norm * 2 * np.pi)).double().cuda()

        op_layer = odl_torch.operator.OperatorModule(operator)
        op_layer_adjoint = odl_torch.operator.OperatorModule(operator.adjoint)

        fbp = odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9) * np.sqrt(2)
        op_layer_fbp = odl_torch.operator.OperatorModule(fbp)

        return op_layer, op_layer_adjoint, op_layer_fbp, op_norm

    def _load_sino_file(self, img_path):
        """
        支持读取:
        - png
        - npy

        最终返回:
        sino_np: shape = [D, A] = [367, 36]
        """
        if img_path.endswith(".npy"):
            sino_np = np.load(img_path).astype(np.float32)

            # 兼容 [1, D, A] / [D, A, 1]
            if sino_np.ndim == 3:
                if sino_np.shape[0] == 1:
                    sino_np = sino_np[0]
                elif sino_np.shape[-1] == 1:
                    sino_np = sino_np[..., 0]
                else:
                    raise ValueError(f"{img_path} 的 npy 维度异常: {sino_np.shape}")

            if sino_np.ndim != 2:
                raise ValueError(f"{img_path} 的 npy 不是二维数组: {sino_np.shape}")

        else:
            sino_img = Image.open(img_path).convert('L')
            sino_np = np.array(sino_img).astype(np.float32)

        return sino_np

    def __getitem__(self, idx):
        # 1. 获取路径
        img_path = self.file_list[idx]

        # 2. 读取 png / npy
        sino_np = self._load_sino_file(img_path)

        # 3. 检查尺寸
        if sino_np.shape != (self.num_detectors, self.num_view):
            raise ValueError(
                f"{img_path} 的尺寸是 {sino_np.shape}, "
                f"但期望是 ({self.num_detectors}, {self.num_view})"
            )

        # 4. 转 tensor 并放到 GPU
        sample = torch.from_numpy(sino_np).float().cuda()   # [D, A]

        # 5. 归一化
        sample = (sample - sample.min()) / (sample.max() - sample.min() + 1e-8)

        # 6. 补通道，变成 [C, D, A]
        sample = sample.unsqueeze(0).cuda()   # [1, D, A]
        channel_size = sample.shape[0]

        # 7. 为单样本创建输出张量
        sino_noisy_sample = torch.zeros(
            (channel_size, self.num_detectors, self.num_view),
            dtype=torch.float32,
            device=sample.device
        )
        fbp_u_sample = torch.zeros(
            (channel_size, 256, 256),
            dtype=torch.float32,
            device=sample.device
        )

        # 8. 逐通道处理
        for j in range(channel_size):
            sino = sample[j]   # [D, A]

            # 调整成 ODL FBP 需要的格式 [1,1,A,D]
            sino_for_fbp = sino.transpose(1, 0).unsqueeze(0).unsqueeze(0)

            # FBP 重建
            fbp_u = self.fbp_curr(sino_for_fbp).squeeze(0)   # [1,256,256]

            # 保存
            sino_noisy_sample[j] = sino
            fbp_u_sample[j] = fbp_u.squeeze().transpose(1, 0)

        # 返回：重建 CT 图，原始正弦图
        return fbp_u_sample, sino_noisy_sample




# 新增：单张正弦图路径直接测试函数
def test_single_sino(file_path, num_view=36, num_detectors=367, save_name="./fbp_single.png"):
    # 初始化算子（随便给个空目录也行，只要不读文件列表）
    dummy_dir = os.path.dirname(file_path)
    ct_provider = CTSlice_Provider(dummy_dir, num_view=num_view, num_detectors=num_detectors)
    
    # 手动加载单张
    sino_np = ct_provider._load_sino_file(file_path)
    
    # 校验尺寸
    if sino_np.shape != (num_detectors, num_view):
        raise ValueError(f"输入尺寸{sino_np.shape}，要求({num_detectors},{num_view})")
    
    # 转GPU tensor
    sample = torch.from_numpy(sino_np).float().cuda()
    sample = (sample - sample.min()) / (sample.max() - sample.min() + 1e-8)
    sample = sample.unsqueeze(0)  # [1, D, A]

    # FBP重建
    sino_for_fbp = sample.transpose(2, 1).unsqueeze(0).unsqueeze(0)
    fbp_u = ct_provider.fbp_curr(sino_for_fbp).squeeze()
    fbp_u = fbp_u.transpose(1, 0)

    # 转图像保存
    ct_img = fbp_u.detach().cpu().numpy()
    ct_img = (ct_img - ct_img.min()) / (ct_img.max() - ct_img.min() + 1e-8)
    ct_img = (ct_img * 255).astype(np.uint8)
    Image.fromarray(ct_img).save(save_name)

    print(f"重建完成，shape: {fbp_u.shape}，已保存为 {save_name}")
    return fbp_u, sample


# 生成单张 CT 图测试
# 直接调用：填你的单张文件路径即可
if __name__ == '__main__':
    print('Single Sino Test Start')

    # 只需要改这一行路径
    single_npy_path = "/root/code/YJNet/data/dataset_THZ_uncert/test/sino_gt_npy/3_31_7.npy"
    
    # 运行测试并保存
    test_single_sino(
        file_path=single_npy_path,
        num_view=36,
        num_detectors=367,
        save_name="./fbp_result.png"
    )

