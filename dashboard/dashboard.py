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

# Configure OpenCV settings for cloud deployment
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def is_cloud_environment():
    """Check if running in Streamlit Cloud"""
    return os.getenv('STREAMLIT_RUNTIME_ENVIRONMENT') == 'cloud'


def extract_landmarks(image, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
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


def initialize_camera():
    """Initialize camera with better error handling"""
    try:
        if is_cloud_environment():
            st.warning(
                "⚠️ Running in cloud environment. Camera access might be limited.")

        # Try different camera indices
        for index in [0, 1]:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                return cap
            cap.release()

        st.error("""
        🎥 Camera not available. 
        
        If you're running this on Streamlit Cloud:
        - Camera access is limited in cloud environments
        - For full functionality, please run the app locally
        
        If you're running locally:
        - Make sure your camera is connected and not in use by another application
        - Try granting camera permissions to your browser
        """)
        return None

    except Exception as e:
        st.error(f"""
        ❌ Error initializing camera: {str(e)}
        
        System Info:
        - OS: {platform.system()}
        - Python: {platform.python_version()}
        - OpenCV: {cv2.__version__}
        """)
        return None


def main():
    st.title('Real-Time BISINDO Classification')

    if 'camera' not in st.session_state:
        st.session_state.camera = None
        st.session_state.camera_initialized = False

    if is_cloud_environment():
        st.warning("""
        ⚠️ Note: You're running this app in Streamlit Cloud.
        Some features like camera access might be limited.
        For full functionality, consider running the app locally.
        """)

    st.sidebar.header('Choose a Model (My Future Work)')
    model = st.sidebar.selectbox(
        'Select Model', ['RF_BISINDO_99'], disabled=True)

    st.sidebar.header('Webcam Feed Settings')
    brightness = st.sidebar.slider('Brightness', -100, 100, 0)
    contrast = st.sidebar.slider('Contrast', -100, 100, 0)
    saturation = st.sidebar.slider('Saturation', -100, 100, 0)

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

    # Camera controls
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    prediction_text = st.empty()

    if run:
        if not st.session_state.camera_initialized:
            st.session_state.camera = initialize_camera()
            st.session_state.camera_initialized = True

        if st.session_state.camera is not None:
            while run:
                try:
                    ret, frame = st.session_state.camera.read()
                    if not ret:
                        st.error("Failed to get frame from camera")
                        break

                    frame = cv2.flip(frame, 1)
                    frame = cv2.convertScaleAbs(
                        frame, alpha=1 + contrast/100, beta=brightness)

                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    hsv[..., 1] = cv2.add(hsv[..., 1], saturation)
                    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                    landmarks, hand_landmarks = extract_landmarks(
                        frame,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence
                    )

                    if landmarks:
                        landmarks_np = np.array(landmarks).reshape(1, -1)
                        prediction = clf.predict(landmarks_np)
                        predicted_label = prediction[0]

                        cv2.putText(frame, f'Sign: {predicted_label}', (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        for hand_landmark in hand_landmarks:
                            mp_drawing.draw_landmarks(frame, hand_landmark,
                                                      mp_hands.HAND_CONNECTIONS)

                        prediction_text.text(
                            f"Predicted Sign: {predicted_label}")

                    FRAME_WINDOW.image(frame, channels='BGR')
                except Exception as e:
                    st.error(f"Error processing frame: {str(e)}")
                    break
    else:
        if st.session_state.camera is not None:
            st.session_state.camera.release()
            st.session_state.camera = None
            st.session_state.camera_initialized = False

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
