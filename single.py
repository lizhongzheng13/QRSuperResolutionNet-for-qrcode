import torch
# import argparse # <--- 移除 argparse
from PIL import Image
import numpy as np
from torchvision import transforms
from pyzbar.pyzbar import decode
import os
import time

# 假设你的模型定义在 'model.py' 文件中
# 确保 model.py 与此脚本在同一目录下或在 Python 路径中
try:
    from model import QRSuperResolutionNet
except ImportError:
    print("错误：无法导入模型定义。请确保 'model.py' 文件存在且包含 QRSuperResolutionNet 类。")
    exit()


def preprocess_image(image_path, device):
    """加载并预处理图像"""
    try:
        img = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"错误：输入图片未在 {image_path} 找到")
        return None
    except Exception as e:
        print(f"打开图片时出错: {e}")
        return None
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor


def postprocess_image(tensor):
    """将模型输出的 Tensor 转换回 PIL 图像"""
    img_np = tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np, mode='L')
    return pil_img

# main 函数保持不变，它期望接收一个包含配置属性的对象 (之前是 args)


def main(config):  # <--- 函数签名不变，接收一个配置对象
    # --- 0. 设置设备 ---
    if config.use_cpu:  # <--- 使用 config.use_cpu
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # --- 1. 加载模型 ---
    if not os.path.exists(config.weights):  # <--- 使用 config.weights
        print(f"错误：模型权重文件未在 {config.weights} 找到")
        return

    model = QRSuperResolutionNet(
        in_channels=1,
        out_channels=1,
        base_channels=config.base_channels,  # <--- 使用 config.base_channels
        num_blocks=config.num_blocks      # <--- 使用 config.num_blocks
    )
    try:
        model.load_state_dict(torch.load(
            config.weights, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print("模型加载成功。")
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        print("请检查权重文件是否与模型结构匹配（通道数、块数等）。")
        return

    # --- 2. 预处理输入图像 ---
    print(f"正在加载和预处理输入图像: {config.input}")  # <--- 使用 config.input
    lr_tensor = preprocess_image(config.input, device)
    if lr_tensor is None:
        return
    print(f"输入图像张量尺寸: {lr_tensor.shape}")

    # --- 3. 执行超分辨率 ---
    print("开始执行超分辨率处理...")
    start_time = time.time()
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    end_time = time.time()
    print(f"超分辨率处理完成。耗时: {end_time - start_time:.4f} 秒")
    print(f"输出图像张量尺寸: {sr_tensor.shape}")

    # --- 4. 后处理输出图像 ---
    sr_image_pil = postprocess_image(sr_tensor)

    # --- 5. 尝试解码二维码 ---
    print("正在尝试解码超分辨率处理后的二维码...")
    try:
        decoded_objects = decode(sr_image_pil)
    except Exception as e:
        print(f"使用 pyzbar 解码时出错: {e}")
        decoded_objects = []

    if decoded_objects:
        print("-" * 30)
        print("🎉 成功解码二维码！")
        for i, obj in enumerate(decoded_objects):
            try:
                data = obj.data.decode('utf-8')
            except UnicodeDecodeError:
                data = obj.data
            print(f"  结果 {i+1}:")
            print(f"    类型: {obj.type}")
            print(f"    数据: {data}")
        print("-" * 30)
    else:
        print("-" * 30)
        print("❌ 未能从超分辨率图像中解码二维码。")
        print("-" * 30)

    # --- 6. (可选) 保存输出图像 ---
    if config.output:  # <--- 使用 config.output
        try:
            output_dir = os.path.dirname(config.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            sr_image_pil.save(config.output)
            print(f"超分辨率图像已保存到: {config.output}")
        except Exception as e:
            print(f"保存输出图像时出错: {e}")

    # --- 7. (可选) 保存原始尺寸对比图 ---
    if config.compare:  # <--- 使用 config.compare
        try:
            lr_img = Image.open(config.input).convert('L')
            bicubic_img = lr_img.resize(sr_image_pil.size, Image.BICUBIC)
            total_width = bicubic_img.width + sr_image_pil.width
            max_height = max(bicubic_img.height, sr_image_pil.height)
            comparison_img = Image.new('L', (total_width, max_height))
            comparison_img.paste(bicubic_img, (0, 0))
            comparison_img.paste(sr_image_pil, (bicubic_img.width, 0))
            compare_path = os.path.splitext(config.output)[
                0] + "_compare.png" if config.output else "comparison.png"
            comparison_img.save(compare_path)
            print(f"对比图像 (左: Bicubic, 右: SR) 已保存到: {compare_path}")
        except Exception as e:
            print(f"创建或保存对比图像时出错: {e}")


# 定义一个简单的类来存储配置，模拟 argparse 的 Namespace 对象
class Config:
    pass


if __name__ == "__main__":
    # --- 直接在此处设置参数 ---
    config = Config()  # 创建一个空对象来存储配置

    # **必须** 修改下面的路径为你自己的文件路径
    # <--- *** EDIT THIS PATH ***
    # config.input = "/root/autodl-tmp/for_me/dataset/test/lr/qr_286.png"
    config.input = "/root/autodl-tmp/for_me/dataset_3_yasuo/test/lr/qr_768.png"
    # <--- *** EDIT THIS PATH ***
    config.weights = "/root/autodl-tmp/for_me/mine_model_v1/checkpoints/qr_sr_epoch30.pth"

    # **可选** 修改下面的参数
    config.output = "output_sr.png"             # 输出文件名
    config.compare = True                       # 是否保存对比图 (True 或 False)
    config.base_channels = 64                   # 模型的 base_channels (必须与权重匹配)
    config.num_blocks = 5                       # 模型的 num_blocks (必须与权重匹配)
    config.use_cpu = False                      # 是否强制使用 CPU (True 或 False)
    # --- 参数设置结束 ---

    # 检查文件是否存在 (可选但推荐)
    if not os.path.exists(config.input):
        print(f"错误：输入文件未找到: {config.input}")
    elif not os.path.exists(config.weights):
        print(f"错误：权重文件未找到: {config.weights}")
    else:
        # 使用设置好的 config 对象调用 main 函数
        main(config)
