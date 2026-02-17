# backend/get_model.py
import tensorflow as tf
import os

print("正在下载 MobileNetV2 预训练模型，请稍候...")

# 1. 下载在 ImageNet 上预训练过的 MobileNetV2 模型
# include_top=True 表示我们需要它完整的分类能力（能识别1000种物体）
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

# 2. 保存模型到本地
save_path = 'pet_model.h5'
model.save(save_path)

print(f"模型已成功下载并保存为: {os.path.abspath(save_path)}")
