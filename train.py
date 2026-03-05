'''
==============================================================================
项目名称：基于 MobileNetV2 的宠物品种（犬类）识别模型训练脚本
==============================================================================


1. 输入：
   读取 224x224 的彩色宠物图片（包含 R, G, B 三个通道）。
2. 预处理：
   使用 MobileNetV2 专属的 preprocess_input 函数，将像素值归一化到 [-1, 1] 之间，以匹配预训练模型的输入分布。
3. 特征提取 (Backbone)：
   采用在 ImageNet 上预训练的 MobileNetV2 作为底座。
   策略：冻结底座权重（不进行微调），仅利用其强大的基础视觉特征提取能力，大幅降低显存占用并加快训练速度。
4. 推理决策 (Head)：
   - GlobalAveragePooling2D：将高维特征图压缩为一维向量，减少参数。
   - Dense + Dropout(0.5)：自定义的全连接层，并加入 50% 随机失活防止过拟合。
   - Softmax：输出层，将网络信号转换为各个犬种的概率分布（总和为1）。
5. 输出与映射：
   训练完成后生成 pet_mobilenet_model_dog.h5 模型，并自动保存类别索引到 class_indices.txt。
   后端 API 依靠此索引将预测的数字标签映射为英文品种，再查字典翻译为中文返回给小程序。

核心优化策略:
- 强力数据增强 (Data Augmentation)：应对小样本学习，抑制过拟合。
- 动态学习率衰减 (ReduceLROnPlateau)：遇到精度瓶颈时自动降低学习率。
- 早停机制 (EarlyStopping)：防止无效训练，自动保存最佳权重。
'''



import os
# 1. 硬件设置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 屏蔽一些无关紧要的日志
cuda_bin_path = r'E:\Anaconda3\envs\grad_project\Library\bin'
if os.path.exists(cuda_bin_path):
    os.environ['PATH'] = cuda_bin_path + os.pathsep + os.environ['PATH']

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
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
        print(f"正在使用 GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("未发现 GPU，将使用 CPU 训练")


# 2. 参数设置
img_width, img_height = 224, 224
batch_size = 8
epochs = 30

# 路径修复：使用绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.abspath(os.path.join(current_dir, '..', 'dataset', 'train1'))
print(f"数据集绝对路径: {train_dir}")


# 3. 数据增强与加载
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
print(f"检测到 {num_classes} 个类别。")

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


# 4. 构建模型 (MobileNetV2 - 冻结底座)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# 冻结底座，只训练我们新加的头部
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x) # 增加模型复杂度
x = Dropout(0.5)(x)                  # 防过拟合
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])


# 5. 训练

# 早停：如果验证集 5 轮不涨，就停
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
# 学习率衰减：如果 3 轮不涨，学习率减半
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, verbose=1)

print("\n开始训练 MobileNetV2")
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stop, reduce_lr]
)

# 保存
model.save('pet_mobilenet_model_dog.h5')
print("模型已保存为 pet_mobilenet_model_dog.h5")

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('MobileNetV2 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('acc_mobilenet_dog.png')
plt.show()