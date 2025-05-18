<div align="right">
  <a href="README.md">Switch to English</a>
</div>

# QRSuperResolutionNet：用于二维码图像超分辨率的深度学习模型

本项目旨在通过深度学习方法，将低质量二维码图像恢复为高质量图像，并提升其扫码识别成功率。模型结构融合了 RRDB、SEBlock 与 TransformerBlock，并采用感知损失与可识别性损失联合优化，取得了良好的实验效果。

<img src="https://lzz-1340752507.cos.ap-shanghai.myqcloud.com/lzz/compare.png" alt="compare" style="zoom:33%;" />

<img src="https://lzz-1340752507.cos.ap-shanghai.myqcloud.com/lzz/image-20250518100429171.png" alt="image-20250518100429171" style="zoom:50%;" />

---

### 一、项目特点

- **输入/输出**：将 64×64 单通道灰度二维码图像重建为 256×256 高质量图像
- **模块集成**：
  - RRDB：保留高频细节
  - SEBlock：通道注意力机制
  - TransformerBlock：捕获全局上下文信息
- **损失函数**：
  - L1 损失
  - 感知损失（VGG 特征）
  - 识别损失（pyzbar 扫码结果）

---

### 二、项目目录结构

```yaml
├── checkpoints/ # 模型权重保存目录
├── vis/ # 可视化结果输出目录
├── dataset.py # 数据集加载模块
├── model.py # QRSuperResolutionNet 模型定义
├── unused_model.py # 备用模型实现
├── train.py # 训练脚本
├── test.py # 批量测试入口
├── single.py # 单张图像超分辨率测试
├── to_64_photo.py # 图像缩小至 64x64 工具脚本
├── output_sr.png # 示例：超分辨图像输出
├── output_sr_compare.png # 示例：对比图像输出
├── qr_accuracy_results.txt # 扫码识别准确率记录
├── qr_network # 已保存的模型结构（可能为 pkl 或权重）
├── unused_test_normal.py # 备用测试脚本
├── pycache/ # 缓存目录
├── README_zh.md # 中文说明文档
└── README_en.md # 英文说明文档
```

###  三、使用方法

首先注意，数据集的格式需要为

```tex
dataset
├── train
|		└──lr
|		└──hr
└── test
		└──lr
		└──hr
```



##### 1.模型训练

```python
python train.py
```

训练过程中模型权重会自动保存在 `checkpoints/` 目录。

训练自己的模型需要修改代码中的路径位置 。

例如：

```python
train_dataset = QRSRDataset(
        "../dataset/train/lr", "../dataset/train/hr")
```

##### 2.，模型测试

训练自己的模型同样需要修改代码中的路径位置 。

批量测试：

```python
python test.py
```



单图测试：

```python
python single.py
```

### 四、可视化与结果分析

- `output_sr.png`：重建图像示例
- `output_sr_compare.png`：输入、输出与真实图像对比
- `qr_accuracy_results.txt`：记录模型在测试集上的二维码识别准确率



### 五、其他工具

- `to_64_photo.py`：将高分辨率图像降采样为 64×64，用于生成训练输入数据
- `unused_*.py`：备用测试/模型脚本，不参与主流程

### 六、讨论与贡献

欢迎大家积极讨论、反馈与贡献，共同推动二维码超分辨率技术的发展与应用。ヾ(≧▽≦*)o(❁´◡`❁)

