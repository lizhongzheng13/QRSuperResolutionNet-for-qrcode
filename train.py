import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from pyzbar.pyzbar import decode
import easyocr
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from dataset import QRSRDataset
from model import QRSuperResolutionNet
# from model_test import QRSuperResolutionNet

# 感知损失（使用 VGG 提取图像特征）


class PerceptualLoss(nn.Module):
    def __init__(self, vgg_model):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg_model
        self.criterion = nn.MSELoss()

    def forward(self, sr_img, hr_img):
        sr_rgb = sr_img.repeat(1, 3, 1, 1)  # 将单通道图像扩展为 RGB
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
        self.reader = easyocr.Reader(['en'])  # 初始化 easyocr 识别器

    def forward(self, sr_imgs):
        loss = 0.0
        sr_imgs = sr_imgs.detach().cpu()
        for img in sr_imgs:
            pil_img = Image.fromarray(
                (img.squeeze().numpy() * 255).astype(np.uint8), mode='L')
            if not self.is_recognizable(pil_img):
                loss += 1.0  # 增加惩罚
        return loss

    def is_recognizable(self, pil_img):
        # 使用 pyzbar 解码
        pyzbar_result = decode(pil_img)
        if pyzbar_result:
            return True
        return False


# 主训练函数
def run():
    batch_size = 8
    epochs = 50
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    # train_dataset = QRSRDataset("../dataset/train/lr", "../dataset/train/hr")
    train_dataset = QRSRDataset(
        "../dataset/train/lr", "../dataset/train/hr")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 模型定义
    model = QRSuperResolutionNet().to(device)
    vgg_model = VGG16().to(device).eval()

    # 损失函数 & 优化器
    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss(vgg_model).to(device)
    recognizability_loss = RecognizabilityLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")

        for batch_idx, (lr_imgs, hr_imgs) in enumerate(loop):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            sr_imgs = model(lr_imgs)

            # 各类损失
            loss_l1 = l1_loss(sr_imgs, hr_imgs)
            loss_perceptual = perceptual_loss(sr_imgs, hr_imgs)
            loss_recognizable = recognizability_loss(sr_imgs)

            # 总损失（你可调节权重）
            total_loss = loss_l1 + 0.1 * loss_perceptual + 0.2 * loss_recognizable

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 显示指标
            epoch_loss += total_loss.item()
            loop.set_postfix(l1=loss_l1.item(), perceptual=loss_perceptual.item(),
                             ssim=calc_ssim(sr_imgs, hr_imgs), total=total_loss.item())

            # 保存图片
            if epoch % 1 == 0 and batch_idx == 0:
                os.makedirs("vis", exist_ok=True)
                save_image(
                    lr_imgs, f"vis/epoch{epoch}_lr.png", nrow=4, normalize=True)
                save_image(
                    sr_imgs, f"vis/epoch{epoch}_sr.png", nrow=4, normalize=True)
                save_image(
                    hr_imgs, f"vis/epoch{epoch}_hr.png", nrow=4, normalize=True)

        print(
            f"✅ Epoch {epoch} finished. Avg Loss: {epoch_loss / len(train_loader):.6f}")

        # 计算准确率
        acc = evaluate_recognition_accuracy(model, train_loader, device)
        print(f"Recognition Accuracy at Epoch {epoch}: {acc}%")

        # 保存模型
        if epoch % 5 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(),
                       f"checkpoints/qr_sr_epoch{epoch}.pth")

        scheduler.step()


def calc_ssim(sr_imgs, hr_imgs):
    sr = sr_imgs[0].detach().cpu().numpy()
    hr = hr_imgs[0].detach().cpu().numpy()

    # 确保图像有变化，避免标准差为零
    if np.all(sr == sr[0]) or np.all(hr == hr[0]):  # 如果图像是常数，返回 0
        return 0.0

    min_dim = min(sr.shape[0], sr.shape[1])
    win_size = min(7, min_dim)  # 设置窗口大小不超过最小维度的 7 或者图像尺寸

    # 计算 SSIM 时指定 data_range
    return ssim(sr, hr, win_size=win_size, data_range=1)


def evaluate_recognition_accuracy(model, dataloader, device):
    model.eval()
    total = 0
    recognized = 0

    with torch.no_grad():
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs = lr_imgs.to(device)
            sr_imgs = model(lr_imgs).cpu()

            for img in sr_imgs:
                img_np = (img.squeeze().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np, mode='L')
                if len(decode(pil_img)) > 0:
                    recognized += 1
                total += 1

    acc = recognized / total if total > 0 else 0
    return round(acc * 100, 2)


if __name__ == "__main__":
    run()
