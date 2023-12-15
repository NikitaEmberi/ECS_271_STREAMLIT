import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
import librosa
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.utils import to_categorical
import tensorflow as tf
from keras_multi_head import MultiHead
import gdown

# Define the Google Drive file ID
file_id = '10Yn3BVQa2CkjyVFzYkLPqHyjtDnNAAvz'

# Define the output file name
output_file = 'SER_model_3.h5'

# Construct the download link
url = f'https://drive.google.com/uc?id={file_id}'

# Download the file
gdown.download(url, output_file, quiet=False)

model = tf.keras.models.load_model(output_file, custom_objects={'MultiHead': MultiHead})


def shifting(data,rate=1000):
    augmented_data=int(np.random.uniform(low=-5,high=5)*rate)
    augmented_data=np.roll(data,augmented_data)
    return augmented_data

def pitch(data,sr,pitch_factor=0.7,random = False):
    if random:
        pitch_factor=np.random.random() * pitch_factor
    return librosa.effects.pitch_shift(data,sr=sr, n_steps=pitch_factor)

def stretching(data,rate=0.8):
    return librosa.effects.time_stretch(data,rate =rate)

def extract_features(data,sampling_rate):

    result = np.array([])

    #Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T,axis=0)
    result = np.hstack((result,zcr))

    #mfcc
    mfcc = np.mean(librosa.feature.mfcc(y=data,sr = sampling_rate).T,axis=0)
    result = np.hstack((result,mfcc))

    #root mean square val
    rms = np.mean(librosa.feature.rms(y=data).T,axis=0)
    result = np.hstack((result,rms))

    #MelSpectogram
    melspectogram = np.mean(librosa.feature.melspectrogram(y=data,sr = sampling_rate).T, axis=0)
    result = np.hstack((result,melspectogram))

    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files.
    data, sampling_rate = librosa.load(path,duration = 2.5, offset =0.6)

    audio1 = extract_features(data,sampling_rate)
    result = np.array(audio1)

    # data with stretching and pitching
    stretched_audio = stretching(data)
    pitched_audio = pitch(stretched_audio,sampling_rate)
    audio3 = extract_features(pitched_audio,sampling_rate)
    result = np.vstack((result,audio3))


    return result


def main():
    st.title("VoiceVibe: Unleashing Emotion with Speech Emotion Recognition")
    st.write(f"""
    Virtual Assistants and chatbots have significantly narrowed the gap between humans and AI. Through the
    integration of Human-in-the Loop techniques and Human-Computer Interaction (HCI) policies, these assistants are
    now able to answer human needs. ChatGPT stands as a prominent example. offering responses to all questions,
    while there are other virtual assistants that operate for a specific cause (e.g., Counseling Services). Our focus is on
    improving and making online counseling services more accessible to everyone. Recognizing that not everyone can
    articulate their emotions in text. To better understand the emotional state of the person by analyzing voice tone,
    pitch, and intonation, we propose the use of Speech Emotion Recognition (SER) integration with the assistant. In
    this application, the person can verbally communicate with the assistant and ask for help. Using SER, the assistant
    can understand the person's state and offer appropriate feedback. This will further help build trust between humans
    and AI, which is a crucial principle in HCI.
    """)


    st.write("In order to recognize the emotion, Upload the speech audio here: ")

    encoding_dictionary ={0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    uploaded_file = st.file_uploader("", type=["wav"], key='unique_id_1')
    if uploaded_file is not None:
      st.audio(uploaded_file, format='audio/wav')
      feature = get_features(uploaded_file)
      pred_data = model.predict(feature)
      encoder = OneHotEncoder()
      pred_labels_encoded = np.argmax(pred_data, axis=1)
      st.write(f"Predicted Label: {encoding_dictionary[pred_labels_encoded[0]]}")

    st.write("##### You can take sample audio files from here: https://drive.google.com/drive/folders/1QrVXYcsf5moqNT0dGig-PN9OitbFCeWQ?usp=sharing")

    st.write("#### Group Members:", markdown=True)



    members_info = [
    {"Name": "Taher Travadi", "Photo": "https://i.postimg.cc/Qxt91zLD/Taher.jpg", "Email": "ttravadi@ucdavis.edu", "LinkedIn": "https://www.linkedin.com/in/taher-travadi/"},
    {"Name": "Nikita B. Emberi", "Photo": "https://i.postimg.cc/QNKBFBvb/Nikita-ML.jpg", "Email": "nemberi@ucdavis.edu", "LinkedIn": "https://www.linkedin.com/in/nikitaemberi/"},
    {"Name": "Savali Deshmukh", "Photo": "https://i.postimg.cc/jSYwysy7/savali-ML.jpg", "Email": "sdeshmukh@ucdavis.edu", "LinkedIn": "https://www.linkedin.com/in/savali-d-2092611a6/"},
    ]

    # Create three columns
    col1, col2, col3 = st.columns(3)

    # Insert information into each column
    for member_info, col in zip(members_info, [col1, col2, col3]):
        with col:
            st.subheader(member_info["Name"])
            st.image(member_info['Photo'], caption=member_info['Name'], use_column_width=True)
            st.write(f"Email: {member_info['Email']}")
            url = member_info["LinkedIn"]
            st.write("linkedIn: [link](%s)" % url)

    footer = """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #2c3e50; /* Dark gray background color */
            color: #ecf0f1; /* Light text color */
            text-align: center;
            padding: 10px;
        }
    </style>
    <div class="footer">
        <p>ECS 271 Project. CopyRight: All Rights Reserved.</p>
    </div>
    """

    # Display the footer using the st.markdown function
    st.markdown(footer, unsafe_allow_html=True)






if __name__=='__main__':
  main()
