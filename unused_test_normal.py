# import os
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision.utils import save_image
# from PIL import Image
# import numpy as np
#
# from dataset import QRSRDataset
# from model import QRSuperResolutionNet
# from torch.optim.lr_scheduler import StepLR
# from torchvision.models import vgg16, VGG16_Weights
# from pyzbar.pyzbar import decode
#
#
# # 感知损失（使用 VGG 提取图像特征）
# class PerceptualLoss(nn.Module):
#     def __init__(self, vgg_model):
#         super(PerceptualLoss, self).__init__()
#         self.vgg = vgg_model
#         self.criterion = nn.MSELoss()
#
#     def forward(self, sr_img, hr_img):
#         sr_rgb = sr_img.repeat(1, 3, 1, 1)
#         hr_rgb = hr_img.repeat(1, 3, 1, 1)
#         sr_feat = self.vgg(sr_rgb)
#         hr_feat = self.vgg(hr_rgb)
#         return self.criterion(sr_feat, hr_feat)
#
#
# # VGG16 前 16 层作为感知特征提取器
# class VGG16(nn.Module):
#     def __init__(self):
#         super(VGG16, self).__init__()
#         vgg_model = vgg16(weights=VGG16_Weights.DEFAULT).features
#         self.vgg_layers = nn.Sequential(*list(vgg_model.children())[:16])
#
#     def forward(self, x):
#         return self.vgg_layers(x)
#
#
# # 识别损失（识别失败就加惩罚）
# class RecognizabilityLoss(nn.Module):
#     def __init__(self):
#         super(RecognizabilityLoss, self).__init__()
#
#     def forward(self, sr_imgs):
#         loss = 0.0
#         sr_imgs = sr_imgs.detach().cpu()
#         for img in sr_imgs:
#             img_np = (img.squeeze().cpu().numpy() * 255).astype(np.uint8)
#             pil_img = Image.fromarray(img_np, mode='L')
#             if len(decode(pil_img)) == 0:
#                 loss += 1.0
#         return torch.tensor(loss / len(sr_imgs), requires_grad=False)
#
#
# # SSIM 简易计算（你可以换更精确的实现）
# def calc_ssim(sr_imgs, hr_imgs):
#     sr = sr_imgs[0].detach().cpu().numpy()
#     hr = hr_imgs[0].detach().cpu().numpy()
#     # 假设为灰度图，shape: [1, H, W]
#     ssim = np.mean((2 * sr * hr + 1e-4) / (sr ** 2 + hr ** 2 + 1e-4))
#     return round(ssim, 3)
#
#
# # 测试函数
# def test(model, test_loader, device):
#     model.eval()
#     total_loss = 0.0
#     total_ssim = 0.0
#     total_recognized = 0
#     total_images = 0
#
#     l1_loss = nn.L1Loss().to(device)
#     perceptual_loss = PerceptualLoss(VGG16().to(device)).to(device)
#     recognizability_loss = RecognizabilityLoss().to(device)
#
#     with torch.no_grad():
#         for lr_imgs, hr_imgs in test_loader:
#             lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
#             sr_imgs = model(lr_imgs)
#
#             # 计算损失
#             loss_l1 = l1_loss(sr_imgs, hr_imgs)
#             loss_perceptual = perceptual_loss(sr_imgs, hr_imgs)
#             loss_recognizable = recognizability_loss(sr_imgs)
#
#             # 计算总损失
#             total_loss += loss_l1.item() + 0.1 * loss_perceptual.item() + \
#                 0.2 * loss_recognizable.item()
#
#             # 计算 SSIM
#             total_ssim += calc_ssim(sr_imgs, hr_imgs)
#
#             # 计算二维码识别准确率
#             for img in sr_imgs:
#                 img_np = (img.squeeze().cpu().numpy() * 255).astype(np.uint8)
#                 pil_img = Image.fromarray(img_np, mode='L')
#                 if len(decode(pil_img)) > 0:
#                     total_recognized += 1
#                 total_images += 1
#
#     avg_loss = total_loss / len(test_loader)
#     avg_ssim = total_ssim / len(test_loader)
#     avg_recognized = total_recognized / total_images if total_images > 0 else 0
#     accuracy = avg_recognized * 100
#
#     print(
#         f"Test Loss: {avg_loss:.6f}, Test SSIM: {avg_ssim:.3f}, Test QR Code Accuracy: {accuracy:.2f}%")
#
#
# # 主函数
# def run():
#     batch_size = 16
#     lr = 1e-4
#     epochs = 30
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 加载测试集
#     test_dataset = QRSRDataset("../dataset/test/lr", "../dataset/test/hr")
#     test_loader = DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#
#     # 加载训练好的模型
#     model = QRSuperResolutionNet().to(device)
#     model.load_state_dict(torch.load(
#         "checkpoints/qr_sr_epoch20.pth"))  # 加载最后一个 epoch 的模型
#
#     # 执行测试
#     test(model, test_loader, device)
#
#
# if __name__ == "__main__":
#     run()
