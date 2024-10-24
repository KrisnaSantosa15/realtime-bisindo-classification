import os
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import platform

import warnings
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def is_cloud_environment():
    """Check if running in Streamlit Cloud"""
    return os.getenv('STREAMLIT_RUNTIME_ENVIRONMENT') == 'cloud'


def extract_landmarks(image, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            if len(results.multi_hand_landmarks) == 1:
                landmarks.extend([0] * 63)
            return landmarks, results.multi_hand_landmarks
    return None, None


def main():
    st.title('BISINDO Classification (Static Image)')

    st.warning(
        """Please clone this [project](https://github.com/KrisnaSantosa15/realtime-bisindo-classification) and run dashboard.py locally to use realtime webcam features.""")

    st.sidebar.header('Choose a Model (My Future Work)')
    model = st.sidebar.selectbox(
        'Select Model', ['RF_BISINDO_99'], disabled=True)

    st.sidebar.header('Model Settings')
    min_detection_confidence = st.sidebar.slider(
        'Min Detection Confidence', 0.0, 1.0, 0.5)
    min_tracking_confidence = st.sidebar.slider(
        'Min Tracking Confidence', 0.0, 1.0, 0.5)

    try:
        model_path = f'model/{model.lower()}.pkl'
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            return
        clf = joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return

    st.sidebar.header('Image Upload')
    uploaded_image = st.sidebar.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"])

    predefined_images = os.listdir('images/example')
    selected_image = st.sidebar.selectbox(
        'Or choose a pre-defined image', predefined_images)

    if uploaded_image is not None:
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
    else:
        image_path = os.path.join('images/example', selected_image)
        image = cv2.imread(image_path)

    prediction_header = st.empty()

    if image is not None:
        landmarks, hand_landmarks = extract_landmarks(
            image,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        if landmarks:
            landmarks_np = np.array(landmarks).reshape(1, -1)
            prediction = clf.predict(landmarks_np)
            predicted_label = prediction[0]

            prediction_header.header(f"Predicted Sign: {predicted_label}")

            st.image(image, channels="BGR",
                     caption=f'Predicted Sign: {predicted_label}')

            for hand_landmark in hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmark, mp_hands.HAND_CONNECTIONS)

        else:
            st.warning("No hand landmarks detected. Try another image.")

    with st.container():
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            st.markdown("### 👨‍💻 Project Information")
            st.info("""
            **Created by: Krisna Santosa**

            This is an original work for BISINDO (Indonesian Sign Language) Classification.

            If you want to modify or use this code, please provide proper attribution.

            I am very open to any feedback or suggestions. Feel free to contact me on [LinkedIn](https://www.linkedin.com/in/krisna-santosa/). One more, Let's collaborate!
            """)
        st.markdown("---")


if __name__ == "__main__":
    main()
