# yolo
import time
import cv2
import os
import numpy as np
import time as t

# mediapipe

import pickle
import mediapipe as mp


def yolo_cam(frame, my_nms_threshold):#tiny-yolov3による検出を行う関数
    yolo_info = [0] * 5 #printデータを保存するリスト

    nms_threshold = my_nms_threshold
    with open("./classes.txt", "r") as f:#classファイルをロード. classファイルに記述されている物体のみ検知可能
        class_names = [cname.strip() for cname in f.readlines()]
    net = cv2.dnn.readNet("model_yolo.weights", "model_yolo.cfg") #重みファイルをロード

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    model = cv2.dnn_DetectionModel(net) #モデルのインスタンス生成
    model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True) #入力画像のサイズを指定

    classes, scores, boxes = model.detect(frame, 0.7, nms_threshold) #classes:検出された物体名, scores:確実度, boxes:boundary boxの座標

    yolo_info[0] = len(boxes)
    for (classid, score, box) in zip(classes, scores, boxes):
        color = (255, 255, 0)
        label = "%s : %f" % (class_names[classid], score)
        cv2.rectangle(frame, box, color, 2) #boundary boxの描画
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)#テキスト挿入

    cv2.imshow('YOLO-Tiny3', frame)
    cv2.waitKey(1)#1を引数に入れた場合表示が連続的になる

    if yolo_info[0] != 0:#print用リストにデータを代入
        yolo_info[1] = box
        yolo_info[2] = culc_pos(box)
        yolo_info[3] = classes
        yolo_info[4] = score
    else:
        print("no person")#人が検出されなかった場合

    return yolo_info


def culc_pos(box_data):#baundary boxの重心を計算する関数
    cornerX = box_data[0]
    cornerY = box_data[1]
    box_width = box_data[2]
    box_height = box_data[3]

    position = [cornerX + 0.5 * box_width, cornerY + 0.5 * box_height]

    return position


def culc_size(box_data):#baundary boxの面積を計算する関数
    size = np.sqrt(box_data[2] * box_data[3])
    return size


def uav(classe_data, score_data, box_data):#自律制御を想定した関数: 未完成・非実装
    boundary_size_big = 30
    boundary_size_small = 20

    if score_data == 0 and box_data == 0:
        print("no person")
    else:
        if classe_data == 0 and score_data > 0.4:
            if culc_size(box_data) > boundary_size_big:
                print("close")
                # back
            elif boundary_size_big > culc_size(box_data) > boundary_size_small:
                print("moderate")
                # stay
            elif culc_size(box_data) < boundary_size_small:
                print("far")
                # forward


def detect_hands(hands_model, draw_hands=0):#手を検知する関数
    result_hands = [0] * 4 #print用データを保存するリスト

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    hand_dict = {0: 'one', 1: 'three', 2: 'two'}

    model = hands_model

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
            model_complexity=0,
            max_num_hands=1,#検知できる手の数の最大値
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():  # while cam is accessible
            success, image = cap.read()

            result_hands[0] = 0
            if not success:  # when there is error with cam
                print("Ignoring empty camera frame.")
                result_hands[0] = 1
                continue  # break from this loop

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # read image by opencv
            results = hands.process(image)  # process by mediapipe

            mark_list = []

            if results.multi_hand_landmarks:  # if hands are detected
                for hand_landmark in results.multi_hand_landmarks:  # get data from multi_hand_landmarks
                    for i in range(21):
                        x = hand_landmark.landmark[i].x
                        y = hand_landmark.landmark[i].y
                        z = hand_landmark.landmark[i].z

                        mark_list.append(x)
                        mark_list.append(y)
                        mark_list.append(z)

                mark_list = np.array(mark_list)  # create numpy array

                if len(mark_list) > 1:  # when there is data
                    detect_flag = 1
                    result_hands[1] = 0
                else:  # when there is no data
                    detect_flag = 0
                    result_hands[1] = 1
            else:  # if hands are not detected
                detect_flag = 0
                result_hands[1] = 1

            if detect_flag == 1:  # when hands are detected and there is data

                prediction = model.predict(mark_list.reshape(1, -1))  # classification result
                prediction_prob = model.predict_proba(mark_list.reshape(1, -1))[0][prediction[0]]  # probability

                if prediction == 0:  # one
                    # print(hand_dict[0])
                    result_hands[2] = "one"
                elif prediction == 1:  # three
                    # print(hand_dict[1])
                    result_hands[2] = "three"
                elif prediction == 2:  # two
                    # print(hand_dict[2])
                    result_hands[2] = "two"

                # print(prediction_prob)
                result_hands[3] = prediction_prob

            else:
                print("No hands")

            # Draw the hand annotations on the image.

            if draw_hands == 0:
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

                if cv2.waitKey(1) or 0xFF == 27:
                    break
    cap.release()
    return result_hands


if __name__ == '__main__':
    with open('./logistic2.pkl', 'rb') as f:
        my_model = pickle.load(f)

    _yolo_info = [0] * 5
    _hands_info = [0] * 4

    capture = cv2.VideoCapture(0)

    phase = 0

    while True:
        if phase == 0:
            start = t.time()

            while True:
                capture = cv2.VideoCapture(0)
                ret, frame = capture.read()
                img_cam = cv2.resize(frame, (560, 480))
                _yolo_info = yolo_cam(img_cam, 0.4)
                print(_yolo_info)

                if _yolo_info[0] == 0:
                    start = t.time()

                end = t.time()

                if end - start > 10:
                    phase = 1
                    capture.release()
                    cv2.destroyAllWindows()
                    break
                else:
                    phase = 0

        elif phase == 1:
            start = t.time()
            capture = cv2.VideoCapture(0)
            while True:
                ret, frame = capture.read()
                img_cam = cv2.resize(frame, (560, 480))
                _yolo_info = yolo_cam(img_cam, 0.4)
                print(_yolo_info)

                if _yolo_info[0] == 0:
                    end = t.time()
                else:
                    end = start

                if end - start > 10:
                    phase = 0
                    break

                box_size = culc_size(_yolo_info[1])
                print(box_size)

                distance_coefficient = 450*50

                distance = distance_coefficient * (1/box_size)  # 距離をcmで求める

                distance_phase = 0

                if distance > 150:
                    print("Forward")
                    distance_phase = 0
                # 前進
                elif 150 >= distance >= 50:
                    print("stay")
                    distance_phase = 1
                # 待機
                elif distance < 50:
                    print("back")
                    distance_phase = 2
                # 後退

                print(distance)
                end = t.time()

                if distance_phase == 1:
                    phase = 2
                    break

            capture.release()
            cv2.destroyAllWindows()
        # yolo v3単体の試験をしたい場合はこれ以下をコメントアウト
        elif phase == 2:
            while True:
                _hands_info = detect_hands(my_model, 0)
                print(_hands_info)
