'''
训练脚本：使用迁移学习和强力数据增强，构建一个高效的图像分类模型。
    输入： 用户上传一张 224x224 的彩色图片（三个颜色通道 R, G, B）。

    预处理： 像素值被缩放到 0 到 1 之间，方便计算机计算。

    特征提取： MobileNetV2 层层过滤，把图片变成抽象的特征信号。

    推理决策： 全连接层和 Dropout 层处理这些信号，最后通过 Softmax 转换成概率。

    输出： 返回一个数字索引（如 2），后端 app.py 根据 class_indices.txt 查到 2 代表 husky，再通过 API 翻译成“哈士奇”发给小程序。
'''
# 毕业设计：基于深度学习与迁移学习的宠物品种识别模型训练脚本
# 架构：MobileNetV2 (特征提取) + GlobalAveragePooling2D + Dense (分类推理)
# 优化：GPU混合精度加速、数据增强(防过拟合)、早停机制、两阶段微调


import os
# ==========================================
# 1. 硬件设置 (尝试调用 GPU，失败则回退 CPU)
# ==========================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 屏蔽一些无关紧要的日志
cuda_bin_path = r'E:\Anaconda3\envs\grad_project\Library\bin'
if os.path.exists(cuda_bin_path):
    os.environ['PATH'] = cuda_bin_path + os.pathsep + os.environ['PATH']

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
# 【关键】导入 MobileNetV2 专用的预处理函数
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt



gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ 正在使用 GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ 未发现 GPU，将使用 CPU 训练 (速度较慢，请耐心等待)")

# ==========================================
# 2. 参数设置
# ==========================================
img_width, img_height = 224, 224
# MobileNet 比较小，CPU 也能跑得动 32，如果内存不够就改 16
batch_size = 32
# 不微调的情况下，我们可以多跑几轮，让顶层充分收敛
epochs = 30

# 路径修复：使用绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假设你的文件夹叫 dataset/train (或者 train1，请根据实际情况修改)
train_dir = os.path.abspath(os.path.join(current_dir, '..', 'dataset', 'train1'))
print(f"📍 数据集绝对路径: {train_dir}")

# ==========================================
# 3. 数据增强与加载
# ==========================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # 使用 MobileNet 官方预处理 (-1到1)
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

# 自动获取并排序类别
all_folders = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
my_sorted_classes = sorted(all_folders, key=str.lower)
num_classes = len(my_sorted_classes)
print(f"✅ 检测到 {num_classes} 个类别。")

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='categorical',
    subset='training', classes=my_sorted_classes, shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    train_dir, target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='categorical',
    subset='validation', classes=my_sorted_classes, shuffle=False
)

# 保存索引
with open('class_indices.txt', 'w', encoding='utf-8') as f:
    f.write(str(my_sorted_classes))

# ==========================================
# 4. 构建模型 (MobileNetV2 - 冻结底座)
# ==========================================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# 【核心】冻结底座，不进行微调
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x) # 增加模型复杂度
x = Dropout(0.5)(x)                  # 防过拟合
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# ==========================================
# 5. 训练
# ==========================================
# 早停：如果验证集 5 轮不涨，就停
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
# 学习率衰减：如果 3 轮不涨，学习率减半 (帮助突破瓶颈)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, verbose=1)

print("\n🚀 开始训练 MobileNetV2 (无微调模式)...")
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stop, reduce_lr]
)

# 保存
model.save('pet_mobilenet_model.h5')
print("✅ 模型已保存为 pet_mobilenet_model.h5")

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('MobileNetV2 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('acc_mobilenet.png')
plt.show()