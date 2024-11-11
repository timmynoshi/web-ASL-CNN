from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf

# Tải mô hình
model = tf.keras.models.load_model("model_level_3.keras")
label = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'blank']

app = Flask(__name__)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 128, 128, 1)  # Thay đổi kích thước để phù hợp với mô hình của bạn
    return feature / 255.0

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Vẽ hình chữ nhật xung quanh khu vực cần crop
        cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)

        # Lấy vùng ảnh để dự đoán
        crop_frame = frame[40:300, 0:300]  # Crop vùng từ frame
        crop_frame_gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
        crop_frame_resized = cv2.resize(crop_frame_gray, (128, 128))
        crop_frame_features = extract_features(crop_frame_resized)

        # Dự đoán với mô hình đã tải
        pred = model.predict(crop_frame_features)
        prediction_label = label[pred.argmax()]
        confidence = f"{np.max(pred) * 100:.2f}%"

        # Hiển thị kết quả dự đoán trên khung hình
        cv2.putText(frame, f"{prediction_label.upper()} {confidence}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        # Encode hình ảnh thành JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
