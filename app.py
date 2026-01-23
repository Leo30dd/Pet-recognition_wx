from flask import Flask, request, jsonify
from flask_cors import CORS    # 允许跨域请求
import time

app = Flask(__name__)
CORS(app)                     # 初始化跨域

@app.route("/")
def home():
    return "Hello 嘉嘉"

@app.route("/predict", methods =['POST'])
def predict():
    print("接收到请求......")   # 在控制台打印日志
    # 判断有无文件
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400  # 返回错误响应
    # 获取文件对象
    file = request.files.get('file')
    # 空文件检查
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400  # 返回错误响应
    # 读取文件内容
    if file:     # 确保文件存在
        print(f"成功接受文件：{file.filename}")
        # 模拟耗时
        time.sleep(2)

        # 模拟返回预测结果
        fake_result = {
            "filename": file.filename,
            "prediction": "cat",   # 假设预测结果为“cat”
            "confidence": 0.95     # 假设置信度为95%
        }

        return jsonify(fake_result), 200   # 返回成功响应


if __name__ == "__main__":
    app.run(debug=True)