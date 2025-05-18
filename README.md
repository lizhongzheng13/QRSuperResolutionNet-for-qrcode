<div align="right">
  <a href="README_zh.md">Switch to Chinese</a>
</div>

# QRSuperResolutionNet: A Deep Learning Model for QR Code Super-Resolution

This project aims to restore low-quality QR code images to high-quality ones using deep learning methods, improving their scanning recognition success rate. The model architecture integrates RRDB, SEBlock, and TransformerBlock, and is jointly optimized with perceptual loss and recognizability loss, achieving good experimental results.

<img src="https://lzz-1340752507.cos.ap-shanghai.myqcloud.com/lzz/compare.png" alt="compare" style="zoom:33%;" />

<img src="https://lzz-1340752507.cos.ap-shanghai.myqcloud.com/lzz/image-20250518100429171.png" alt="image-20250518100429171" style="zoom:50%;" />

---

### 1. Project Features

- **Input/Output**: Reconstruct 64×64 single-channel grayscale QR code images into 256×256 high-quality images
- **Module Integration**:
  - RRDB: preserves high-frequency details
  - SEBlock: channel attention mechanism
  - TransformerBlock: captures global contextual information
- **Loss Functions**:
  - L1 loss
  - Perceptual loss (VGG features)
  - Recognizability loss (pyzbar scan results)

---

### 2. Project Directory Structure

```yaml
├── checkpoints/ # Directory for saving model weights
├── vis/ # Directory for visualization outputs
├── dataset.py # Dataset loading module
├── model.py # QRSuperResolutionNet model definition
├── unused_model.py # Alternative/backup model implementation
├── train.py # Training script
├── test.py # Batch testing entry script
├── single.py # Single image super-resolution testing
├── to_64_photo.py # Tool script to downscale images to 64x64
├── output_sr.png # Example: super-resolved image output
├── output_sr_compare.png # Example: comparison image output
├── qr_accuracy_results.txt # Record of QR code recognition accuracy
├── qr_network # Saved model architecture (possibly pkl or weights)
├── unused_test_normal.py # Backup testing script
├── pycache/ # Cache directory
├── README_zh.md # Chinese README
└── README_en.md # English README
```



### 3. Usage Instructions

Please note, the dataset format should be:

```tex
dataset
├── train
|    └── lr
|    └── hr
└── test
     └── lr
     └── hr
```

##### 1. Model Training

```python
python train.py
```

During training, model weights will be automatically saved in the `checkpoints/` directory.

You need to modify the dataset paths in the code to match your environment.

For example:

```python
train_dataset = QRSRDataset(
        "../dataset/train/lr", "../dataset/train/hr")
```

##### 2. Model Testing

Similarly, modify the dataset paths for testing your own model.

Batch testing:

```python
python test.py
```

Single image testing:

```python
python single.py
```

### 4. Visualization and Result Analysis

- `output_sr.png`: Example of reconstructed images
- `output_sr_compare.png`: Comparison of input, output, and ground truth images
- `qr_accuracy_results.txt`: Records QR code recognition accuracy on the test set

------

### 5. Other Tools

- `to_64_photo.py`: Downscales high-resolution images to 64×64 for generating training inputs
- `unused_*.py`: Backup test/model scripts not involved in the main pipeline

---

### 6. Discussion and Contributions

We warmly welcome everyone to actively discuss, provide feedback, and contribute to this project to help improve QR code super-resolution research and applications.ヾ(≧▽≦*)o(❁´◡`❁)





