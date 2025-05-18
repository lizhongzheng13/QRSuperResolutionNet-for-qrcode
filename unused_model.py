# @Author : LiZhongzheng
# 开发时间  ：2025-05-18 9:32

# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super(SEBlock, self).__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y


# class ResidualDenseBlock(nn.Module):
#     def __init__(self, channels=64, growth_channels=32):
#         super().__init__()
#         self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1)
#         self.conv2 = nn.Conv2d(channels + growth_channels,
#                                growth_channels, 3, 1, 1)
#         self.conv3 = nn.Conv2d(
#             channels + 2 * growth_channels, growth_channels, 3, 1, 1)
#         self.conv4 = nn.Conv2d(
#             channels + 3 * growth_channels, growth_channels, 3, 1, 1)
#         self.conv5 = nn.Conv2d(
#             channels + 4 * growth_channels, channels, 3, 1, 1)
#         self.lrelu = nn.LeakyReLU(0.2, inplace=True)

#     def forward(self, x):
#         x1 = self.lrelu(self.conv1(x))
#         x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
#         x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
#         x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
#         x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
#         return x + 0.2 * x5


# class RRDB(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.rdb1 = ResidualDenseBlock(channels)
#         self.rdb2 = ResidualDenseBlock(channels)
#         self.rdb3 = ResidualDenseBlock(channels)
#         self.se = SEBlock(channels)

#     def forward(self, x):
#         out = self.rdb1(x)
#         out = self.rdb2(out)
#         out = self.rdb3(out)
#         out = x + 0.2 * out
#         out = self.se(out)
#         return out


# class TransformerBlock(nn.Module):
#     def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.1):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(
#             dim, num_heads, dropout=dropout, batch_first=True)
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, int(dim * mlp_ratio)),
#             nn.ReLU(inplace=True),
#             nn.Linear(int(dim * mlp_ratio), dim)
#         )

#     def forward(self, x):
#         b, c, h, w = x.shape
#         x_flat = x.view(b, c, -1).permute(0, 2, 1)  # B x N x C
#         x_norm = self.norm1(x_flat)
#         attn_out, _ = self.attn(x_norm, x_norm, x_norm)
#         x = x_flat + attn_out
#         x = x + self.mlp(self.norm2(x))
#         x = x.permute(0, 2, 1).view(b, c, h, w)
#         return x


# class QRSuperResolutionNet(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, base_channels=64, num_blocks=5):
#         super().__init__()
#         self.entry = nn.Conv2d(in_channels, base_channels, 3, 1, 1)

#         # 主体 RRDB 模块
#         self.body = nn.Sequential(*[RRDB(base_channels)
#                                   for _ in range(num_blocks)])

#         # Transformer 编码模块
#         self.transformer = TransformerBlock(dim=base_channels)

#         # 上采样跳跃分支
#         self.skip_up = nn.Sequential(
#             nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False),
#             nn.Conv2d(in_channels, out_channels, 3, 1, 1)
#         )

#         # PixelShuffle 上采样
#         self.upsample = nn.Sequential(
#             nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
#             nn.PixelShuffle(2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
#             nn.PixelShuffle(2),
#             nn.LeakyReLU(0.2, inplace=True)
#         )

#         self.exit = nn.Conv2d(base_channels, out_channels, 3, 1, 1)

#     def forward(self, x):
#         feat = self.entry(x)
#         feat = self.body(feat)
#         feat = self.transformer(feat)  # 加入 transformer 结构
#         feat = self.upsample(feat)
#         out = self.exit(feat)

#         # 融合 Bicubic 分支输出
#         skip = self.skip_up(x)
#         out = out + skip

#         return torch.clamp(out, 0.0, 1.0)


# # 测试模型尺寸
# if __name__ == "__main__":
#     model = QRSuperResolutionNet()
#     dummy_input = torch.randn(1, 1, 64, 64)
#     output = model(dummy_input)
#     print("输出尺寸：", output.shape)  # 预期：(1, 1, 256, 256)
# # @Author : LiZhongzheng
# # 开发时间  ：2025-04-30 21:34


# class ResidualDenseBlock(nn.Module):
#     def __init__(self, channels=64, growth_channels=32):
#         super().__init__()
#         self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1)
#         self.conv2 = nn.Conv2d(channels + growth_channels,
#                                growth_channels, 3, 1, 1)
#         self.conv3 = nn.Conv2d(
#             channels + 2 * growth_channels, growth_channels, 3, 1, 1)
#         self.conv4 = nn.Conv2d(
#             channels + 3 * growth_channels, growth_channels, 3, 1, 1)
#         self.conv5 = nn.Conv2d(
#             channels + 4 * growth_channels, channels, 3, 1, 1)
#         self.lrelu = nn.LeakyReLU(0.2, inplace=True)

#     def forward(self, x):
#         x1 = self.lrelu(self.conv1(x))
#         x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
#         x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
#         x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
#         x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
#         return x + 0.2 * x5


# class RRDB(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.rdb1 = ResidualDenseBlock(channels)
#         self.rdb2 = ResidualDenseBlock(channels)
#         self.rdb3 = ResidualDenseBlock(channels)

#     def forward(self, x):
#         out = self.rdb1(x)
#         out = self.rdb2(out)
#         out = self.rdb3(out)
#         return x + 0.2 * out


# class QRSuperResolutionNet(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, num_blocks=5, base_channels=64):
#         super().__init__()
#         self.conv_first = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
#         self.rrdb_blocks = nn.Sequential(
#             *[RRDB(base_channels) for _ in range(num_blocks)])
#         self.trunk_conv = nn.Conv2d(base_channels, base_channels, 3, 1, 1)

#         # 上采样 ×4（2x2 次）
#         self.upsample = nn.Sequential(
#             nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
#             nn.PixelShuffle(2),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
#             nn.PixelShuffle(2),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#         self.conv_last = nn.Conv2d(base_channels, out_channels, 3, 1, 1)

#     def forward(self, x):
#         # 第一层卷积
#         fea = self.conv_first(x)

#         # RRDB块
#         trunk = self.trunk_conv(self.rrdb_blocks(fea))
#         fea = fea + trunk  # 残差连接

#         # 上采样
#         out = self.upsample(fea)

#         # 最后一层卷积
#         out = self.conv_last(out)

#         # 使用sigmoid确保输出在 [0, 1] 范围
#         return torch.sigmoid(out)  # 这里修改为sigmoid处理输出


# # 测试模型尺寸
# if __name__ == "__main__":
#     model = QRSuperResolutionNet()
#     dummy_input = torch.randn(1, 1, 64, 64)  # Batch=1, 单通道, 64x64
#     output = model(dummy_input)
#     print("输出尺寸：", output.shape)  # 应该是 (1, 1, 256, 256)