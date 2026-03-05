import os
import ast
import uuid
from datetime import datetime
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import pymysql
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# 兼容性设置
pymysql.install_as_MySQLdb()

app = Flask(__name__)
CORS(app)

#  数据库配置
DB_PASSWORD = "zcl2004..."
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://root:{DB_PASSWORD}@127.0.0.1:3306/pet_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JSON_AS_ASCII'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

db = SQLAlchemy(app)


#  数据模型
class History(db.Model):
    __tablename__ = 'history_records'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(64), index=True, nullable=False)
    image_url = db.Column(db.String(255), nullable=False)
    breed = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    create_time = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self, host_url):
        return {
            'id': self.id,
            'breed': self.breed,
            'confidence': f"{self.confidence:.1f}",
            'image_url': host_url.rstrip('/') + self.image_url,
            'time': self.create_time.strftime("%Y-%m-%d %H:%M")
        }


with app.app_context():
    db.create_all()

#  模型加载
MODEL_PATH = 'pet_mobilenet_model_dog.h5'
CLASS_INDICES_PATH = 'class_indices.txt'
CLASS_NAMES = []
model = None

# 字典
PET_NAMES_MAP = {
    'beagle': '比格犬', 'border_collie': '边境牧羊犬', 'chihuahua': '吉娃娃',
    'chow': '松狮', 'collie': '柯利牧羊犬','doberman': '杜宾犬', 'french_bulldog': '法国斗牛犬', 'german_shepherd': '德国牧羊犬',
    'golden_retriever': '金毛寻回犬', 'husky': '哈士奇', 'labrador_retriever': '拉布拉多',
    'malamute': '阿拉斯加雪橇犬', 'pembroke': '柯基犬', 'pomeranian': '博美犬',
    'poodle': '贵宾犬', 'pug': '巴哥犬', 'samoyed': '萨摩耶',
    'shiba_dog': '柴犬'
}


def init_resources():
    global model, CLASS_NAMES
    try:
        if os.path.exists(CLASS_INDICES_PATH):
            with open(CLASS_INDICES_PATH, 'r', encoding='utf-8') as f:
                CLASS_NAMES = ast.literal_eval(f.read())
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print(" 模型加载成功")
    except Exception as e:
        print(f" 资源加载失败: {e}")


init_resources()


def preprocess_image(image_path):
    img = Image.open(image_path)
    if img.mode != "RGB": img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


# 核心接口
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files: return jsonify({"code": 400, "msg": "No file"}), 400
    file = request.files['file']
    user_id = request.form.get('user_id', 'anonymous')
    if file.filename == '': return jsonify({"code": 400, "msg": "Empty filename"}), 400

    try:
        # 1. 保存图片
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'jpg'
        unique_filename = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(save_path)

        # 2. 预测
        processed_img = preprocess_image(save_path)
        predictions = model.predict(processed_img)
        top_idx = np.argmax(predictions[0])
        label_en = CLASS_NAMES[top_idx]
        confidence = float(predictions[0][top_idx])

        # 阈值判断
        CONFIDENCE_THRESHOLD = 0.70

        if confidence < CONFIDENCE_THRESHOLD:
            breed_cn = "未识别为狗"
        else:
            breed_cn = PET_NAMES_MAP.get(label_en.lower(), label_en)

        # 3. 存入数据库
        web_path = f"/static/uploads/{unique_filename}"
        record = History(
            user_id=user_id, image_url=web_path, breed=breed_cn,
            confidence=round(confidence * 100, 2)
        )
        db.session.add(record)
        db.session.commit()

        return jsonify({
            "code": 200,
            "data": {
                "breed": breed_cn,
                "confidence": round(confidence * 100, 2),
                "image_url": web_path
            }
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"code": 500, "msg": str(e)}), 500


@app.route("/history", methods=["GET"])
def get_history_list():
    user_id = request.args.get('user_id')
    records = History.query.filter_by(user_id=user_id).order_by(History.create_time.desc()).all()
    host_url = request.host_url
    # 统一返回 image_url 字段
    data = [r.to_dict(host_url) for r in records]
    return jsonify({"code": 200, "data": data})


@app.route("/delete_history", methods=["POST"])
def delete_history():
    # 删除单条记录接口
    try:
        data = request.get_json()
        record_id = data.get('id')
        record = History.query.get(record_id)
        if record:
            db.session.delete(record)
            db.session.commit()
            return jsonify({"code": 200, "msg": "已删除"})
        return jsonify({"code": 404, "msg": "记录不存在"})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)})


@app.route("/history/clear", methods=["POST"])
def clear_history():
    # 清空所有接口
    data = request.get_json()
    user_id = data.get('user_id')
    History.query.filter_by(user_id=user_id).delete()
    db.session.commit()
    return jsonify({"code": 200, "msg": "已清空"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)