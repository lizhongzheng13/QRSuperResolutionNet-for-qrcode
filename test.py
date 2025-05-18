import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
import numpy as np

from dataset import QRSRDataset
from model import QRSuperResolutionNet
from torchvision.models import vgg16, VGG16_Weights
from pyzbar.pyzbar import decode


# 感知损失（使用 VGG 提取图像特征）
class PerceptualLoss(nn.Module):
    def __init__(self, vgg_model):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg_model
        self.criterion = nn.MSELoss()

    def forward(self, sr_img, hr_img):
        sr_rgb = sr_img.repeat(1, 3, 1, 1)
        hr_rgb = hr_img.repeat(1, 3, 1, 1)
        sr_feat = self.vgg(sr_rgb)
        hr_feat = self.vgg(hr_rgb)
        return self.criterion(sr_feat, hr_feat)


# VGG16 前 16 层作为感知特征提取器
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg_model = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.vgg_layers = nn.Sequential(*list(vgg_model.children())[:16])

    def forward(self, x):
        return self.vgg_layers(x)


# 识别损失（识别失败就加惩罚）
class RecognizabilityLoss(nn.Module):
    def __init__(self):
        super(RecognizabilityLoss, self).__init__()

    def forward(self, sr_imgs):
        loss = 0.0
        sr_imgs = sr_imgs.detach().cpu()
        for img in sr_imgs:
            img_np = (img.squeeze().cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np, mode='L')
            if len(decode(pil_img)) == 0:
                loss += 1.0
        return torch.tensor(loss / len(sr_imgs), requires_grad=False)


# SSIM 简易计算
def calc_ssim(sr_imgs, hr_imgs):
    sr = sr_imgs[0].detach().cpu().numpy()
    hr = hr_imgs[0].detach().cpu().numpy()
    ssim = np.mean((2 * sr * hr + 1e-4) / (sr ** 2 + hr ** 2 + 1e-4))
    return round(ssim, 3)

# new


def calc_psnr(sr_imgs, hr_imgs):
    sr = sr_imgs[0].detach().cpu().numpy()
    hr = hr_imgs[0].detach().cpu().numpy()
    mse = np.mean((sr - hr) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))


# 测试函数
def test(model, test_loader, device, model_name=""):
    model.eval()
    total_loss = 0.0
    total_ssim = 0.0
    total_recognized = 0
    total_images = 0
    # new
    total_psnr = 0.0

    l1_loss = nn.L1Loss().to(device)
    perceptual_loss = PerceptualLoss(VGG16().to(device)).to(device)
    recognizability_loss = RecognizabilityLoss().to(device)

    with torch.no_grad():
        for lr_imgs, hr_imgs in test_loader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            sr_imgs = model(lr_imgs)

            loss_l1 = l1_loss(sr_imgs, hr_imgs)
            loss_perceptual = perceptual_loss(sr_imgs, hr_imgs)
            loss_recognizable = recognizability_loss(sr_imgs)

            total_loss += loss_l1.item() + 0.1 * loss_perceptual.item() + \
                0.2 * loss_recognizable.item()

            total_ssim += calc_ssim(sr_imgs, hr_imgs)
            # new
            total_psnr += calc_psnr(sr_imgs, hr_imgs)
            for img in sr_imgs:
                img_np = (img.squeeze().cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np, mode='L')
                if len(decode(pil_img)) > 0:
                    total_recognized += 1
                total_images += 1
    # new
    avg_psnr = total_psnr / len(test_loader)
    avg_loss = total_loss / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    accuracy = total_recognized / total_images * 100

    # print(f"[{model_name}] Test Loss: {avg_loss:.6f}, SSIM: {avg_ssim:.3f}, QR Code Accuracy: {accuracy:.2f}%")
    print(f"[{model_name}] Test Loss: {avg_loss:.6f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.3f}, QR Code Accuracy: {accuracy:.2f}%")

    return accuracy


# 主函数：对比多个模型
def run():
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载测试集
    test_dataset = QRSRDataset(
        "../dataset/test/lr", "../dataset/test/hr")
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 模型列表（模型名称: 权重路径）
    models_to_test = {
        "QRSuperRes_v1": "checkpoints/qr_sr_epoch30.pth",
        # "QRSuperRes_v2": "checkpoints/qr_sr_epoch30.pth",
        # "ESRGAN_baseline": "checkpoints/esrgan.pth"
    }

    results = {}

    for model_name, ckpt_path in models_to_test.items():
        model = QRSuperResolutionNet().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        acc = test(model, test_loader, device, model_name=model_name)
        results[model_name] = acc

    # 保存结果到文件
    with open("qr_accuracy_results.txt", "w") as f:
        for name, acc in results.items():
            f.write(f"{name}: {acc:.2f}%\n")


if __name__ == "__main__":
    run()
