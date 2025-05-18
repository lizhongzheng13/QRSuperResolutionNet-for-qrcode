# @Author : LiZhongzheng
# 开发时间  ：2025-04-30 21:37

# import os
# from PIL import Image
# from torch.utils.data import Dataset
# import torchvision.transforms as T


# class QRSRDataset(Dataset):
#     def __init__(self, lr_dir, hr_dir):
#         super().__init__()
#         self.lr_dir = lr_dir
#         self.hr_dir = hr_dir
#         self.lr_images = sorted([
#             f for f in os.listdir(lr_dir) if f.endswith('.png')
#         ])
#         self.hr_images = [name.replace("lr_", "hr_") for name in self.lr_images]

#         # 图像预处理：单通道 + 归一化到 [0, 1]
#         self.to_tensor = T.Compose([
#             T.Grayscale(num_output_channels=1),
#             T.ToTensor()
#         ])

#     def __len__(self):
#         return len(self.lr_images)

#     def __getitem__(self, idx):
#         lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
#         hr_path = os.path.join(self.hr_dir, self.hr_images[idx])

#         if not os.path.exists(hr_path):
#             raise FileNotFoundError(f"HR image missing: {hr_path}")

#         # 加载图像并转换为 tensor
#         lr_img = self.to_tensor(Image.open(lr_path).convert("L"))
#         hr_img = self.to_tensor(Image.open(hr_path).convert("L"))

#         # ✅ 尺寸检查
#         if lr_img.shape[-2:] != (64, 64):
#             raise ValueError(f"LR image size must be 64x64, got {lr_img.shape}")
#         if hr_img.shape[-2:] != (256, 256):
#             raise ValueError(f"HR image size must be 256x256, got {hr_img.shape}")

#         return lr_img, hr_img
# @Author : LiZhongzheng
# 开发时间  ：2025-04-30 21:37

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class QRSRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, auto_resize=False):
        """
        :param lr_dir: 低分辨率图像目录（64x64）
        :param hr_dir: 高分辨率图像目录（256x256）
        :param auto_resize: 是否自动 resize 到指定大小（默认 False，尺寸不符时报错）
        """
        super().__init__()
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.auto_resize = auto_resize

        self.lr_images = sorted([
            f for f in os.listdir(lr_dir) if f.endswith('.png')
        ])
        self.hr_images = [name.replace("lr_", "hr_")
                          for name in self.lr_images]

        self.lr_size = (64, 64)
        self.hr_size = (256, 256)

        self.to_tensor = T.Compose([
            T.Grayscale(num_output_channels=1),  # 强制灰度
            T.ToTensor()  # [0, 255] → [0, 1]
        ])

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])

        if not os.path.exists(hr_path):
            raise FileNotFoundError(f"⚠️ HR image not found: {hr_path}")

        lr_img = Image.open(lr_path).convert("L")
        hr_img = Image.open(hr_path).convert("L")

        # 如果 auto_resize 开启，则强制调整尺寸
        if self.auto_resize:
            lr_img = lr_img.resize(self.lr_size, Image.BICUBIC)
            hr_img = hr_img.resize(self.hr_size, Image.BICUBIC)
        else:
            if lr_img.size != self.lr_size:
                raise ValueError(
                    f"❌ LR image size must be {self.lr_size}, got {lr_img.size} at {lr_path}")
            if hr_img.size != self.hr_size:
                raise ValueError(
                    f"❌ HR image size must be {self.hr_size}, got {hr_img.size} at {hr_path}")

        lr_tensor = self.to_tensor(lr_img)
        hr_tensor = self.to_tensor(hr_img)

        return lr_tensor, hr_tensor
