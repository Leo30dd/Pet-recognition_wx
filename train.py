'''
训练脚本：使用迁移学习和强力数据增强，构建一个高效的图像分类模型。
    输入： 用户上传一张 224x224 的彩色图片（三个颜色通道 R, G, B）。

    预处理： 像素值被缩放到 0 到 1 之间，方便计算机计算。

    特征提取： MobileNetV2 层层过滤，把图片变成抽象的特征信号。

    推理决策： 全连接层和 Dropout 层处理这些信号，最后通过 Softmax 转换成概率。

    输出： 返回一个数字索引（如 2），后端 app.py 根据 class_indices.txt 查到 2 代表 husky，再通过 API 翻译成“哈士奇”发给小程序。
'''
import os

# 手动指定 GPU 驱动搜索路径
#  Conda 环境下存放 cudart64_110.dll 等文件的位置
cuda_bin_path = r'E:\Anaconda3\envs\grad_project\Library\bin'

if os.path.exists(cuda_bin_path):
    # 将该路径添加到系统的环境变量 PATH 的最前面
    os.environ['PATH'] = cuda_bin_path + os.pathsep + os.environ['PATH']
    print(f"已手动指向 GPU 驱动目录: {cuda_bin_path}")
else:
    print("警告：未找到指定的 GPU 驱动目录，请检查路径是否正确")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# 1. GPU 显存优化设置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 设置显存按需增长，防止一次性占满显存导致崩溃
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(" 已开启显存按需增长模式")
    except RuntimeError as e:
        print(f" GPU 配置报错: {e}")

print("="*30)
print("正在使用的设备:", " GPU" if gpus else " CPU")
print("="*30)


# 2. 设置参数
img_width, img_height = 224, 224
batch_size = 64
epochs = 30      # 开启了早停机制，可以放心把上限设高
train_dir = os.path.join('..', 'dataset', 'train')


# 3. 数据预处理与【强力数据增强】(防过拟合)
train_datagen = ImageDataGenerator(
    rescale=1./255,           # 归一化
    rotation_range=30,        # 随机旋转角度 (0-30度)
    width_shift_range=0.2,    # 随机水平平移
    height_shift_range=0.2,   # 随机垂直平移
    shear_range=0.2,          # 随机错切变换
    zoom_range=0.3,           # 随机缩放范围
    horizontal_flip=True,     # 随机水平翻转
    brightness_range=[0.8, 1.2], # 随机调整亮度，模拟不同光照
    fill_mode='nearest',      # 填充像素的方法
    validation_split=0.2      # 自动划分 20% 的数据用于验证集
)

# 核心：获取文件夹并按字母顺序排序（忽略大小写），确保与后端翻译逻辑一致
all_folders = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
my_sorted_classes = sorted(all_folders, key=str.lower)
num_classes = len(my_sorted_classes)
print(f" 检测到 {num_classes} 个类别，已完成排序。")

# 加载训练集 程序去文件夹里看，发现有 5 个文件夹，就自动把它们定为 5 个类别，
# 并且按照 my_sorted_classes 的顺序来分配标签（0, 1, 2, ...）。shuffle=True 打乱训练数据，增加模型泛化能力。
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    classes=my_sorted_classes,
    shuffle=True
)

# 加载验证集
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    classes=my_sorted_classes,
    shuffle=False
)

# 保存类别映射关系到 txt 文件，供 app.py 翻译使用
print(f" 正在保存类别索引到 class_indices.txt...")
with open('class_indices.txt', 'w') as f:
    f.write(str(my_sorted_classes))

print("类别索引:", train_generator.class_indices)

# 4. 构建模型 (迁移学习 + 结构优化)
# 加载预训练底座，去掉最顶层的 1000 类分类器(include_top=False)只保留它超强的“视觉能力”。
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# 迁移学习设置 节省算力 锁定底座权重，不让它在初始训练中被破坏
base_model.trainable = False

# 搭建我们自己的“大脑”
x = base_model.output
x = GlobalAveragePooling2D()(x) # 全局平均池化  压缩数据，减少参数，防止模型因为记得太细而产生过拟合。
x = Dense(1024, activation='relu')(x)  # 全连接层，增加模型表达能力
x = Dropout(0.5)(x)             # 【关键】Dropout层，训练时随机关闭50%神经元，防止过拟合
predictions = Dense(num_classes, activation='softmax')(x) # 输出层

model = Model(inputs=base_model.input, outputs=predictions)

# 编译模型：使用较小的学习率进行平稳训练
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# 5. 设置【早停机制】回调
# 如果验证集准确率连续 4 轮没有提升，则提前结束，并自动加载效果最好那一轮的权重
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=4,
    restore_best_weights=True,
    verbose=1
)


# 6. 开始训练
print("\n 开始训练...")
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stop] # 加入早停回调
)


# 7. 保存与可视化
model.save('my_custom_model.h5')
print("\n 训练圆满完成！模型已保存为: my_custom_model.h5")

# 绘制准确率曲线
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Trend')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png')
print("曲线图已保存为 accuracy_plot.png")
plt.show()