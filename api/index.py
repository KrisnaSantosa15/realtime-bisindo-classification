from flask import Flask, Response, render_template, jsonify
import cv2
import mediapipe as mp
import numpy as np
import joblib
from flask_cors import CORS
import base64
import json

app = Flask(__name__)
CORS(app)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

try:
    clf = joblib.load('model/rf_bisindo_99.pkl')
except:
    clf = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        frame_data = request.json['frame']

        encoded_data = frame_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])

                if len(results.multi_hand_landmarks) == 1:
                    landmarks.extend([0] * 63)

                if clf is not None:
                    landmarks_np = np.array(landmarks).reshape(1, -1)
                    prediction = clf.predict(landmarks_np)[0]
                else:
                    prediction = "Model not loaded"

                for hand_landmark in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmark,
                        mp_hands.HAND_CONNECTIONS
                    )

                # Add prediction text
                cv2.putText(
                    frame,
                    f'Sign: {prediction}',
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            _, buffer = cv2.imencode('.jpg', frame)
            processed_frame = base64.b64encode(buffer).decode('utf-8')

            return jsonify({
                'processed_frame': f'data:image/jpeg;base64,{processed_frame}',
                'prediction': prediction if 'prediction' in locals() else None
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
