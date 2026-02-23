
import os
# ==========================================
# 1. 硬件设置
# ==========================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cuda_bin_path = r'E:\Anaconda3\envs\grad_project\Library\bin'
if os.path.exists(cuda_bin_path):
    os.environ['PATH'] = cuda_bin_path + os.pathsep + os.environ['PATH']

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
# 【关键】注意这里换成了 resnet_v2 的预处理
from tensorflow.keras.applications.resnet_v2 import preprocess_input
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
    print("⚠️ 正在使用 CPU 训练 (ResNet 较慢，请挂机等待)")

# ==========================================
# 2. 参数设置
# ==========================================
img_width, img_height = 224, 224
# 【关键】ResNet 很大，为了防崩溃，这里设为 16
batch_size = 16
# 只跑一个阶段，设高一点，让它充分收敛
epochs = 30

current_dir = os.path.dirname(os.path.abspath(__file__))
# 记得核对你的文件夹名字，是 train 还是 train1
train_dir = os.path.abspath(os.path.join(current_dir, '..', 'dataset', 'train1'))
print(f"📍 数据集绝对路径: {train_dir}")

# ==========================================
# 3. 数据增强与加载
# ==========================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # ResNet 专用预处理
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

with open('class_indices.txt', 'w', encoding='utf-8') as f:
    f.write(str(my_sorted_classes))

# ==========================================
# 4. 构建模型 (ResNet50V2 - 冻结底座)
# ==========================================
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# 【核心】冻结底座，不进行微调
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
# ResNet 特征丰富，可以使用更大的全连接层
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# ==========================================
# 5. 训练
# ==========================================
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, verbose=1)

print("\n🚀 开始训练 ResNet50V2 (无微调模式)...")
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stop, reduce_lr]
)

# 保存
model.save('pet_resnet_model.h5')
print("✅ 模型已保存为 pet_resnet_model.h5")

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('ResNet50V2 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('acc_resnet.png')
plt.show()