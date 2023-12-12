from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import base64
#from sklearn.utils.class_weight import compute_class_weight
#from tensorflow.keras.models import load_model
app = Flask(__name__)
prev = 360
prev2 = 360
prev3 = 360
flag = 0
counter=0
stage='up'
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Load the CSV files
correct_df = pd.read_csv('correct.csv')
too_high_df = pd.read_csv('too_high.csv')
too_low_df = pd.read_csv('too_low.csv')

# Add a label column to each dataframe
correct_df['label'] = 'correct'
too_high_df['label'] = 'too_high'
too_low_df['label'] = 'too_low'

data_df = pd.concat([correct_df, too_high_df, too_low_df])

# Shuffle the combined dataframe
data_df = data_df.sample(frac=1).reset_index(drop=True)

# Split the data into features and labels
X = data_df.drop(['label', 'image_id'], axis=1)  # Assuming all other columns are features
y = data_df['label']

# Encode the labels to integers
y = y.map({'correct': 0, 'too_high': 1, 'too_low': 2})

# Train the RandomForest model
forest_model.fit(X, y)

# Load the Keras model
#keras_model = load_model('squat.h5')

# Initialize MediaPipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    angle_min = []
    angle_min_hip = []

    @app.route('/process_frame', methods=['POST'])
    def process_frame():
        global flag
        global prev
        global counter
        global stage

        data = request.get_json()
        base64_image = data.get('frame', '')
        image_data = base64.b64decode(base64_image)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            #mp_drawing.draw_landmarks(
            #    frame,
            #    results.pose_landmarks,
            #    mp_pose.POSE_CONNECTIONS,
            #    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),
            #    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)
            #)

            angle_knee = calculate_angle(hip, knee, ankle)  # Knee joint angle
            angle_knee = round(angle_knee, 2)

            angle_hip = calculate_angle(shoulder, hip, knee)
            angle_hip = round(angle_hip, 2)

            angle_min.append(angle_knee)
            angle_min_hip.append(angle_hip)

            if angle_knee > 150:
                stage="up"
            if angle_knee <=90 and stage=='up':
                stage='down'
                counter+=1
                print(counter)

            if angle_knee > prev and angle_knee<170:
                prediction = get_prediction(angle_knee)
                prediction_result = prediction
                
                #cv2.imwrite("lowest_point.jpg",frame)
            else:
                prediction_result = "Get into position"
            prev = angle_knee

            print(angle_knee,angle_hip)
            results = {
                'prediction': prediction_result,
                'rep':counter
            }
        else:
            results = {
                'prediction': "Get into position",
                'rep':counter
            }
        return jsonify(results)

    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def get_prediction(angle_knee):
        new_data = pd.DataFrame({'angle': [angle_knee]})
        prediction = forest_model.predict(new_data)
        labels = {0: "correct", 1: "too high", 2: "too low"}
        print(labels[prediction[0]])
        return labels[prediction[0]]
        #image = cv2.resize(image, (300, 300))
        #image = image / 255.0
        #image = image.reshape(1, 300, 300, 3)
        # Make predictions using the Keras model
        #prediction = keras_model.predict(image)
        #labels = {0:"correct",1:"too low",2:"too high"}
        #pred_index = np.argmax(prediction[0])
        #print("\nPrediction is: ", labels[pred_index])
        #return labels[pred_index]

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, threaded=True)