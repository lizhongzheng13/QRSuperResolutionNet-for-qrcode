from PIL import Image

# 输入图片路径
input_image_path = '/root/autodl-tmp/for_me/mine_model_v1/image.png'  # 替换为你的图片路径
# 输出图片路径
output_image_path = './resized_image.png'  # 缩放后的图片保存路径

# 打开图片
image = Image.open(input_image_path)

# 将图片缩小到64×64
resized_image = image.resize((64, 64))

# 保存缩小后的图片
resized_image.save(output_image_path)

print(f"图片已缩小到64×64并保存到：{output_image_path}")
