import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def png_to_npy(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    print(f"\n处理目录: {input_dir}")
    print(f"找到 {len(image_files)} 张图像")

    for filename in tqdm(image_files):
        input_path = os.path.join(input_dir, filename)

        image = Image.open(input_path).convert("L")
        
        image = np.array(image).astype(np.float32)

        #splitext把文件切成两段(名字，扩展名)，[0]表示名字，所有功能是改后缀名
        save_name = os.path.splitext(filename)[0] + ".npy"
        save_path = os.path.join(output_dir, save_name)

        np.save(save_path, image)


def batch_png_to_npy(src_root, dst_root):
    sub_dirs = [
        "train/ct_input",
        "train/ct_label",
        "valid/ct_input",
        "valid/ct_label",
        "test/ct_input",
        "test/ct_label",
    ]

    for sub_dir in sub_dirs:
        src_dir = os.path.join(src_root, sub_dir)
        dst_dir = os.path.join(dst_root, sub_dir)

        if not os.path.exists(src_dir):
            print(f"跳过，不存在: {src_dir}")
            continue

        png_to_npy(src_dir, dst_dir)
    
    print("\n全部转换完成")


if __name__ == "__main__":
    batch_png_to_npy(
        src_root="/root/code/YJNet/data/dataset_256_300",
        dst_root="/root/code/YJNet/data/dataset_256_300_npy"
    )




