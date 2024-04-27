import streamlit as st # type: ignore
from model import predict_emotion

# Set the title of the app
st.title("Emotion Prediction from Speech")

# Display the list of emotion categories in a grid layout
st.subheader("Emotion Categories")

# Define the list of emotions
emotions = {1: 'Neutral', 2: 'Happy', 3: 'Sad', 4: 'Angry', 5: 'Fear', 6: 'Disgust', 7: 'Surprise'}

# Create the first row with four columns
col1, col2, col3, col4 = st.columns(4)

# Display the first four emotions in the first row
with col1:
    st.write(f"{1}: {emotions[1]}")
with col2:
    st.write(f"{2}: {emotions[2]}")
with col3:
    st.write(f"{3}: {emotions[3]}")
with col4:
    st.write(f"{4}: {emotions[4]}")

# Create the second row with four columns
col5, col6, col7 = st.columns(3)

# Display the remaining four emotions in the second row
with col5:
    st.write(f"{5}: {emotions[5]}")
with col6:
    st.write(f"{6}: {emotions[6]}")
with col7:
    st.write(f"{7}: {emotions[7]}")

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

