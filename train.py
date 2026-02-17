import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import matplotlib.pyplot as plt

# --- 加入这两行 ---
print("="*30)
print("正在使用的设备:", "GPU" if tf.config.list_physical_devices('GPU') else "CPU")
print("="*30)


# 1. 设置参数
img_width, img_height = 224, 224
batch_size = 16
epochs = 15

# ！！！重要：改成你自己的数据集路径！！！
# 使用 os.path.join 保证跨平台兼容性
# '..' 表示上一级目录，所以是从 backend/ 返回到 MyGraduationProject/ 再进入 dataset/
train_dir = os.path.join('..', 'dataset', 'train')

# 动态获取类别数量
num_classes = len(os.listdir(train_dir))
print(f"发现 {num_classes} 个类别。")

# 2. 数据预处理和增强
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2   # 把20%的数据作为验证集
)

# 获取所有文件夹名
all_folders = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
# 核心：按字母表顺序排序，忽略大小写 (key=str.lower)
my_sorted_classes = sorted(all_folders, key=str.lower)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    classes=my_sorted_classes
)

# 1. 获取类别字典，例如 {'cat': 0, 'dog': 1}
indices_dict = train_generator.class_indices

# 2. 转换成只有名字的列表，并确保顺序正确
# 结果类似: ['cat', 'dog']
class_names = list(indices_dict.keys())

print(f" 检测到类别映射: {indices_dict}")
print(f" 正在保存类别索引到 class_indices.txt...")

'''
# 3. 写入文件
with open('class_indices.txt', 'w') as f:
    f.write(str(class_names))
'''

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    classes=my_sorted_classes
)

# 打印类别索引，非常重要
print("类别索引:", train_generator.class_indices)
# Keras 会按文件夹名的字母顺序排序

# 3. 构建模型（迁移学习）
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

# 4. 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 5. 训练模型
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# 6. 保存模型
model.save('my_custom_model.h5')
print("模型已保存为 backend/my_custom_model.h5")

# 绘制准确率曲线
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png') # 保存图片
plt.show()