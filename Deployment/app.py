import streamlit as st
from model import predict_emotion

# Set the title of the app
st.title("Emotion Prediction from Speech")

# Add a file uploader widget
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac"])

# Check if an audio file has been uploaded
if audio_file is not None:
    # Save the uploaded audio file to a temporary location
    audio_path = 'temp_audio_file'
    with open(audio_path, "wb") as f:
        f.write(audio_file.read())

    # Add an audio player to play the uploaded audio file
    st.audio(audio_file)

    # Add a button to predict emotion
    if st.button("Predict Emotion"):
        # Predict the emotion
        emotion = predict_emotion(audio_path)

        # Display the predicted emotion
        st.write(f"Predicted Emotion: {emotion}")
