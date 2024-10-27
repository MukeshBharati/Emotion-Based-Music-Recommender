import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import webbrowser
import os

# Load the model
try:
    model = tf.keras.models.load_model("saved_model.keras")  # Use .keras if saved in that format
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Load labels
try:
    label = np.load("labels.npy")
    print("Labels loaded successfully.")
except Exception as e:
    print(f"Error loading labels: {e}")

holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("Emotion Based Music Recommender")

if "run" not in st.session_state:
    st.session_state["run"] = True

# Try to load emotion; create a default file if it doesn't exist
if os.path.exists("emotion.npy"):
    try:
        emotion = np.load("emotion.npy")[0]
    except Exception as e:
        print(f"Error loading emotion: {e}")
        emotion = ""
else:
    emotion = ""
    np.save("emotion.npy", np.array([""]))  # Create the file with an empty value

if not emotion:
    st.session_state["run"] = True
else:
    st.session_state["run"] = False

class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        # Ensure that face landmarks are detected
        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            # Process left hand landmarks
            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            # Process right hand landmarks
            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            lst = np.array(lst).reshape(1, -1)

            # Log input shape before prediction
            print(f"Input shape for prediction: {lst.shape}")

            # Prediction
            try:
                pred = label[np.argmax(model.predict(lst))]
                print(f"Detected Emotion: {pred}")
                cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
                np.save("emotion.npy", np.array([pred]))
            except Exception as e:
                print(f"Error during prediction: {e}")

        # Draw landmarks
        if res.face_landmarks is not None:
            drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
                                    connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        if res.left_hand_landmarks is not None:
            drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        if res.right_hand_landmarks is not None:
            drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


lang = st.text_input("Language")
singer = st.text_input("Singer")

if lang and singer and st.session_state["run"]:
    try:
        print("Starting WebRTC streamer...")
        webrtc_streamer(key="key", desired_playing_state=True,
                        video_processor_factory=EmotionProcessor)
        print("WebRTC streamer started.")
    except Exception as e:
        print(f"Error starting WebRTC streamer: {e}")

btn = st.button("Recommend me songs")

if btn:
    if not emotion:
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = True
    else:
        webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = False
