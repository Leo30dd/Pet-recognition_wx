# 文件名: backend/app.py
import os
import ast  # 用于安全地把字符串转换成列表
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # MobileNetV2 专用预处理函数
from tensorflow.keras.preprocessing.image import img_to_array


#  初始化配置
app = Flask(__name__)
CORS(app)

# 1. 模型路径
MODEL_PATH = 'pet_mobilenet_model_dog.h5'  # 确保这是你训练好的模型文件名

# 2. 自动读取类别名称 (替代手动定义)
CLASS_INDICES_PATH = 'class_indices.txt'
CLASS_NAMES = []

# 英汉对照字典
PET_NAMES_MAP = {
    'beagle': '比格犬',
    'border_collie': '边境牧羊犬',
    'chihuahua': '吉娃娃',
    'chow': '松狮',
    'collie': '柯利牧羊犬',
    'doberman': '杜宾犬',
    'french_bulldog': '法国斗牛犬',
    'german_shepherd': '德国牧羊犬',
    'golden_retriever': '金毛寻回犬',
    'husky': '哈士奇',
    'labrador_retriever': '拉布拉多',
    'malamute': '阿拉斯加雪橇犬',
    'pembroke': '柯基犬',
    'pomeranian': '博美犬',
    'poodle': '贵宾犬',            # 泰迪也是贵宾的一种
    'pug': '巴哥犬',
    'samoyed': '萨摩耶',
    'shiba_dog': '柴犬'
}

def load_resources():
    """加载模型和类别文件"""
    global model, CLASS_NAMES

    # 加载类别名称
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, 'r') as f:
            # ast.literal_eval 能把字符串 "['a', 'b']" 安全地变成列表 ['a', 'b']
            CLASS_NAMES = ast.literal_eval(f.read())
        print(f"已加载类别列表: {CLASS_NAMES}")
    else:
        print("错误：找不到 class_indices.txt！请先运行 train.py。")

    # 加载模型
    print(f"正在加载模型 {MODEL_PATH} ...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")


# 启动时加载资源
load_resources()


def process_image(image):
    """图片预处理"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # 归一化
    return img_array


@app.route("/predict", methods=["POST"])
def predict():
    # 1. 基础校验
    if 'file' not in request.files:
        return jsonify({"error": "未接收到文件"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "文件名为空"}), 400

    # 确保模型和类别列表已加载
    if model is None or not CLASS_NAMES:
        return jsonify({"error": "服务端资源未就绪，请检查服务器日志"}), 500

    try:
        # 2. 图片处理
        image = Image.open(file.stream)
        processed_image = process_image(image)

        # 3. 模型预测
        predictions = model.predict(processed_image)

        # 获取概率最大的索引
        predicted_index = np.argmax(predictions[0])

        # 获取原始英文名 (例如: 'American Shorthair' 或 'husky')
        predicted_label_en = CLASS_NAMES[predicted_index]

        # 获取置信度
        confidence = float(predictions[0][predicted_index])

        CONFIDENCE_THRESHOLD = 0.70  # 推荐 0.7

        if confidence < CONFIDENCE_THRESHOLD:
            print(f"非狗类别，置信度过低: {confidence:.2f}")

            return jsonify({
                "breed": "未识别为狗",
                "confidence": round(confidence * 100, 2)
            })

        # 查字典翻译
        # 1. 把名字统一转成小写 (例如: 'American Shorthair' -> 'american shorthair')
        #    这样就能匹配我们在字典里写的小写 Key 了
        lookup_key = predicted_label_en.lower()

        # 2. 查字典，如果查不到，就默认显示原英文名
        breed_cn = PET_NAMES_MAP.get(lookup_key, predicted_label_en)

        print(f" 识别: {predicted_label_en} ->  翻译: {breed_cn} (置信度: {confidence:.2f})")

        # 4. 返回结果
        return jsonify({
            "breed": breed_cn,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        print(f" 预测出错: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)