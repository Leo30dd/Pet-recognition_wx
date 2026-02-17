import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import matplotlib.pyplot as plt


# 1. 设置参数
img_width, img_height = 224, 224
batch_size = 16
epochs = 1

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

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 打印类别索引，非常重要
print("类别索引:", train_generator.class_indices)
# Keras 会按文件夹名的字母顺序排序，例如: {'golden_retriever': 0, 'husky': 1}

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
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# 6. 保存模型
model.save('pet_breed_model.h5')
print("模型已保存为 backend/pet_breed_model.h5")

# 绘制准确率曲线
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png') # 保存图片
plt.show()