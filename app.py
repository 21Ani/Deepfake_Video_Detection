from flask import Flask, render_template, request, jsonify
import os
from main import predict_audio, predict_video

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    if file.filename.endswith(('.wav', '.mp3')):
        label, confidence = predict_audio(file_path)
    elif file.filename.endswith(('.mp4', '.avi', '.mov')):
        label, confidence = predict_video(file_path)
    else:
        return jsonify({"error": "Unsupported file type"})

    return jsonify({"label": label, "confidence": confidence})

if __name__ == '__main__':
    app.run(debug=True)
