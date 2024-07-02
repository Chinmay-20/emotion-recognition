import os
import cv2
import numpy as np
import imutils
from flask import Flask, Response, render_template, redirect, request, url_for
from keras.models import load_model
from werkzeug.utils import secure_filename

# Constants for the emotions
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# Load models
detection_model_path = './haarcascade_frontalface_default.xml'
emotion_model_path = './model.hdf5'
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/live_stream')
def live_stream():
    # This route now renders the live_stream.html template which includes the video feed
    return render_template('live_stream.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_video_stream():
    camera = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            frame = apply_emotion_recognition(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        camera.release()


@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('stream_video', filename=filename))
    return render_template('upload_video.html')


@app.route('/stream_video/<filename>')
def stream_video(filename):
    return render_template('video_stream.html', filename=filename)


@app.route('/processed_video/<filename>')
def processed_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(process_and_stream_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


def process_and_stream_video(video_path):
    cap = cv2.VideoCapture(video_path)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = apply_emotion_recognition(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()


def apply_emotion_recognition(frame):
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi = roi_gray.astype("float") / 255.0
        roi = np.expand_dims(np.expand_dims(roi, axis=-1), axis=0)
        preds = emotion_classifier.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
