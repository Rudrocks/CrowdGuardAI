import cv2
import torch
from flask import Flask, Response
import requests

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Flask app
app = Flask(__name__)

# Threshold for crowd detection
THRESHOLD = 10

# Function to send Telegram alert
def send_telegram_alert(message):
    token = "YOUR_TELEGRAM_BOT_TOKEN"
    chat_id = "YOUR_CHAT_ID"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    requests.post(url, data=data)

# Function to send SMS alert (Twilio)
def send_sms_alert(message):
    # Placeholder: integrate Twilio API here
    print("SMS Alert:", message)

# Video stream generator
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model(frame)
            crowd_count = len(results.xyxy[0])
            if crowd_count > THRESHOLD:
                alert_msg = f"Overcrowding detected! Count: {crowd_count}"
                send_telegram_alert(alert_msg)
                send_sms_alert(alert_msg)

            # Draw bounding boxes
            annotated = results.render()[0]
            ret, buffer = cv2.imencode('.jpg', annotated)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
