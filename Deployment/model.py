import numpy as np
import pickle
from keras.models import model_from_json
from tensorflow.keras.models import Sequential, model_from_json
from keras import layers
from keras.saving import register_keras_serializable
import librosa
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
import tensorflow as tf
import tensorflow.keras.layers as L

model = tf.keras.Sequential([
    L.Conv1D(512,kernel_size=5, strides=1,padding='same', activation='relu',input_shape=(2376,1)),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),
    
    L.Conv1D(512,kernel_size=5,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),
    Dropout(0.2),  # Add dropout layer after the second max pooling layer
    
    L.Conv1D(256,kernel_size=5,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),
    
    L.Conv1D(256,kernel_size=3,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),
    Dropout(0.2),  # Add dropout layer after the fourth max pooling layer
    
    L.Conv1D(128,kernel_size=3,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=3,strides=2,padding='same'),
    Dropout(0.2),  # Add dropout layer after the fifth max pooling layer
    
    L.Flatten(),
    L.Dense(512,activation='relu'),
    L.BatchNormalization(),
    L.Dense(7,activation='softmax')
])
# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')
# model.summary()

# Load the model and tools
def load_model_and_tools():
    # Load the model architecture
    with open('CNN_model.json', 'r') as file:
        model_json = file.read()
    model = model_from_json(model_json)
    
    # Load the model weights
    model.load_weights('CNN_model_weights.h5')
    
    # Load the scaler and encoder
    with open('scalar2.pickle', 'rb') as file:
        scaler = pickle.load(file)
    
    with open('encoder2.pickle', 'rb') as file:
        encoder = pickle.load(file)
    
    return model, scaler, encoder

# Functions for feature extraction
def zcr(data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length))

def rmse(data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.rms(data, frame_length=frame_length, hop_length=hop_length))

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc_features = librosa.feature.mfcc(data, sr=sr, n_mfcc=13)
    return np.squeeze(mfcc_features.T) if not flatten else np.ravel(mfcc_features.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    features = np.hstack([
        zcr(data, frame_length, hop_length),
        rmse(data, frame_length, hop_length),
        mfcc(data, sr, frame_length, hop_length)
    ])
    return features

# Get prediction features
def get_predict_feat(path, scaler):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    features = extract_features(data, sample_rate)
    features_reshaped = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_reshaped)
    final_features = np.expand_dims(features_scaled, axis=2)
    return final_features

# Load the model and tools
model, scaler, encoder = load_model_and_tools()

# Emotion prediction function
def predict_emotion(audio_path):
    # Get features
    features = get_predict_feat(audio_path, scaler)
    
    # Make prediction
    predictions = model.predict(features)
    emotion_label = encoder.inverse_transform(predictions)[0][0]
    
    # Map the predicted label to emotion name
    emotions = {1: 'Neutral', 2: 'Calm', 3: 'Happy', 4: 'Sad', 5: 'Angry', 6: 'Fear', 7: 'Disgust', 8: 'Surprise'}
    emotion = emotions.get(emotion_label, "Unknown")
    
    return emotion
