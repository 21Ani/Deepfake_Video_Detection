import os
import torch
import librosa
from skimage.transform import resize
import numpy as np
import cv2
import tensorflow as tf

# ---------------- Audio Model ----------------
class AudioNet(torch.nn.Module):
    def __init__(self):
        super(AudioNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(64*28*28, 4096)
        self.linear2 = torch.nn.Linear(4096, 1024)
        self.linear3 = torch.nn.Linear(1024, 512)
        self.output = torch.nn.Linear(512, 2)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x); x = self.pool(x)
        x = self.conv2(x); x = self.pool(x)
        x = self.conv3(x); x = self.pool(x)
        x = self.relu(x); x = self.flatten(x)
        x = self.linear1(x); x = self.dropout(x)
        x = self.linear2(x); x = self.dropout(x)
        x = self.linear3(x); x = self.dropout(x)
        x = self.output(x)
        return x

# Load audio model
audio_model = AudioNet()
audio_model.load_state_dict(torch.load("audio_classification_model.pth", map_location='cpu'))
audio_model.eval()

def get_mel_spectrogram(wav_path):
    sr=22050; duration=10; img_height=224; img_width=224
    signal, sr = librosa.load(wav_path, sr=sr, duration=duration)
    spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    spec_resized = resize(spec_db, (img_height,img_width), anti_aliasing=True)
    return spec_resized

def predict_audio(wav_path):
    spectrogram = get_mel_spectrogram(wav_path)
    tensor = torch.tensor(spectrogram).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        output = audio_model(tensor)
        pred = torch.argmax(output, axis=1).item()
    label = 'Real' if pred == 0 else 'Fake'
    confidence = output.softmax(dim=1).numpy()[0][1]*100
    return label, confidence

# ---------------- Video Model ----------------
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 50
CLASSES_LIST = ['Deepfake', 'Original']

def create_conv3d_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv3D(16, (3,3,3), activation='relu', input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        tf.keras.layers.MaxPooling3D(pool_size=(1,2,2), padding='same'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv3D(64, (3,3,3), activation='relu'),
        tf.keras.layers.MaxPooling3D(pool_size=(1,2,2), padding='same'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(CLASSES_LIST), activation='softmax')
    ])
    return model

video_model = create_conv3d_model()
video_model.load_weights("conv3d_model_weights.h5")

def preprocess_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(int(total_frames / SEQUENCE_LENGTH), 1)
    for i in range(SEQUENCE_LENGTH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i*skip)
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))/255.0
        frames.append(frame)
    cap.release()
    if len(frames)==SEQUENCE_LENGTH:
        return np.array(frames)
    return None

def predict_video(video_path):
    frames = preprocess_video(video_path)
    if frames is None:
        return "Error", 0
    input_data = np.expand_dims(frames, axis=0)
    prediction = video_model.predict(input_data)
    pred_index = np.argmax(prediction)
    return CLASSES_LIST[pred_index], prediction[0][pred_index]*100
