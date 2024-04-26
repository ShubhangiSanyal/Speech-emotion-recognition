import streamlit as st
import torch
import torchaudio
import numpy as np
import librosa
from model import EmotionRecognizer
from preprocessing import extract_features

emotions = {0: 'Neutral', 1: 'Calm', 2: 'Happy', 3: 'Sad', 4: 'Angry', 5: 'Fear', 6: 'Disgust', 7: 'Surprise'}
model_path = 'best_model_transformer_2.pth'

@st.cache_resource
def load_model():
    recognizer = EmotionRecognizer(model_path, emotions)
    return recognizer

recognizer = load_model()

st.title('Audio Emotion Recognition')

audio_file = st.file_uploader('Upload an audio file', type=['wav', 'mp3', 'ogg', 'flac'])

if audio_file is not None:
    with st.spinner('Preprocessing audio...'):
        audio_path = 'temp_audio_file'
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())
        # audio_bytes = audio_file.read()
        data, sr = torchaudio.load(audio_path)
        data = data.numpy().squeeze()

        features = extract_features(data, sr)
        features = np.array(features)

    with st.spinner('Recognizing emotion...'):
        predictions = []
        feature = torch.from_numpy(features).float().unsqueeze(0)
        output = recognizer.model(feature)
        pred_idx = output.argmax().item()
        pred_emotion = emotions[pred_idx]
        predictions.append(pred_emotion)

        st.success(f'Predicted emotions: {", ".join(predictions)}')