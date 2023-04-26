import pickle

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hand_dict = {0: 'one', 1: 'three', 2: 'two'}

# one = 0
# two = 2
# three = 1

with open('./model/logistic2.pkl', 'rb') as f:
    model = pickle.load(f)

detect_flag = 0

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():  # webカメラを開く
        success, image = cap.read()
        if not success:  # 何らかの理由でwebカメラにアクセスできなかった場合
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # opencvによる画像の読み込み
        results = hands.process(image)  # mediapipeで画像を処理し,結果をresultsに格納

        # print(results)

        mark_list = []

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                for i in range(21):
                    x = hand_landmark.landmark[i].x
                    y = hand_landmark.landmark[i].y
                    z = hand_landmark.landmark[i].z

                    mark_list.append(x)
                    mark_list.append(y)
                    mark_list.append(z)

            mark_list = np.array(mark_list)

            if len(mark_list) > 1:
                detect_flag = 1
            else:
                detect_flag = 0
        else:
            detect_flag = 0

        # print(mark_list)

        if detect_flag == 1:

            prediction = model.predict(mark_list.reshape(1, -1))
            prediction_prob = model.predict_proba(mark_list.reshape(1, -1))[0][prediction[0]]

            if prediction == 0:
                print(hand_dict[0])
            elif prediction == 1:
                print(hand_dict[1])
            elif prediction == 2:
                print(hand_dict[2])

            print(prediction_prob)

        else:
            print("no hands")

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
