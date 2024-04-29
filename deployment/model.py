import numpy as np
import pickle
from keras.models import model_from_json # type: ignore
import librosa # type: ignore

# Load the model and tools
def load_model_and_tools():
    # Load the model architecture
    with open('CNN_model.json', 'r') as file:
        model_json = file.read()
    model = model_from_json(model_json)
    
    # Load the model weights
    model.load_weights('CNN_model.weights.h5')
    
    # Load the scaler and encoder
    with open('scaler2.pickle', 'rb') as file:
        scaler = pickle.load(file)
    
    with open('encoder2.pickle', 'rb') as file:
        encoder = pickle.load(file)
    
    return model, scaler, encoder

# Functions for feature extraction
def zcr(data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length))

def rmse(data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length))

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc_features = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
    return np.squeeze(mfcc_features.T) if not flatten else np.ravel(mfcc_features.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    features = np.array([])
    features = np.hstack([
        zcr(data, frame_length, hop_length),
        rmse(data, frame_length, hop_length),
        mfcc(data, sr, frame_length, hop_length)
    ])
    return features



def get_predict_feat(path, scaler):
    # Load the audio data
    d, s_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # Extract features from the audio data
    res = extract_features(d)
    result = np.array(res)
    
    # Print the original shape of features
    print("Original shape of features:", result.shape)
    
    # Check if the shape is different from expected (2376)
    expected_shape = 2376
    if result.size != expected_shape:
        # Calculate the difference in size
        size_difference = expected_shape - result.size
        
        # Duplicate the last features to extend the array
        if size_difference > 0:
            # Select the last (2376 - 1620) features
            last_features = result[-size_difference:]
            
            # Extend the result array with the selected last features
            result = np.concatenate((result, last_features))
        
        # If size_difference is negative, truncate the array to the expected shape
        else:
            result = result[:expected_shape]
    
    # Reshape the array to the expected shape (1, 2376)
    result = result.reshape(1, expected_shape)
    
    # Scale the result using the scaler
    i_result = scaler.transform(result)
    
    # Add an extra dimension to the data (to match the model's input shape)
    final_result = np.expand_dims(i_result, axis=2)
    
    return final_result



# Load the model and tools
model, scaler, encoder = load_model_and_tools()

emotions1={1:'Neutral', 2:'Calm', 3:'Happy', 4:'Sad', 5:'Angry', 6:'Fear', 7:'Disgust',8:'Surprise'}
def predict_emotion(path1):
    res=get_predict_feat(path1, scaler)
    predictions=model.predict(res)
    y_pred = encoder.inverse_transform(predictions)
    #print(y_pred[0][0]) 
    return y_pred[0][0]
    
