import threading
import time
import tellopy
import pygame
import cv2
import numpy as np
import av
import pygame.locals
from subprocess import Popen, PIPE
import pyautogui
import os
import sys
import mediapipe as mp
import pickle
import win32com.client as wincl

speak = wincl.Dispatch("SAPI.SpVoice")

from tello-experiment.yolov7_func import *  # 環境に応じてPATHを変更してください
# 人検知には yolov7_func内のyolov7()が用いられています

class JoystickPS3:
    # d-pad
    UP = 4  # UP
    DOWN = 6  # DOWN
    ROTATE_LEFT = 7  # LEFT
    ROTATE_RIGHT = 5  # RIGHT

    # bumper triggers
    TAKEOFF = 11  # R1
    LAND = 10  # L1

    # buttons
    FORWARD = 12  # TRIANGLE
    BACKWARD = 14  # CROSS
    LEFT = 15  # SQUARE
    RIGHT = 13  # CIRCLE

    # axis
    LEFT_X = 0
    LEFT_Y = 1
    RIGHT_X = 2
    RIGHT_Y = 3
    LEFT_X_REVERSE = 1.0
    LEFT_Y_REVERSE = -1.0
    RIGHT_X_REVERSE = 1.0
    RIGHT_Y_REVERSE = -1.0
    DEADZONE = 0.1


class JoystickPS4:
    # d-pad
    UP = 11  # UP
    DOWN = 12  # DOWN
    ROTATE_LEFT = 13  # LEFT
    ROTATE_RIGHT = 14  # RIGHT

    # bumper triggers
    TAKEOFF = 10  # R1
    LAND = 9  # L1
    TAKEPHOTO = 7  # R2
    FLIPR = 6  # L2

    # buttons
    FORWARD = 3  # TRIANGLE
    # FORWARD = 30  # azuma
    BACKWARD = 0  # CROSS
    LEFT = 2  # SQUAREss
    RIGHT = 1  # CIRCLE

    # axis
    LEFT_X = 2
    LEFT_Y = 1
    RIGHT_X = 2
    RIGHT_Y = 3
    LEFT_X_REVERSE = 1.0
    LEFT_Y_REVERSE = -1.0
    RIGHT_X_REVERSE = 1.0
    RIGHT_Y_REVERSE = -1.0
    DEADZONE = 0.08


class JoystickPS4_1:
    # d-pad
    UP = 11  # UP
    DOWN = 12  # DOWN
    ROTATE_LEFT = 13  # LEFT
    ROTATE_RIGHT = 14  # RIGHT

    # bumper triggers
    TAKEOFF = 10  # R1
    LAND = 9  # L1
    TAKEPHOTO = 7  # R2
    FLIPR = 6  # L2

    # buttons
    FORWARD = 3  # TRIANGLE
    BACKWARD = 0  # CROSS
    LEFT = 2  # SQUAREss
    RIGHT = 1  # CIRCLE

    # axis
    LEFT_X = 2
    LEFT_Y = 1
    RIGHT_X = 2
    RIGHT_Y = 3
    LEFT_X_REVERSE = 1.0
    LEFT_Y_REVERSE = -1.0
    RIGHT_X_REVERSE = 1.0
    RIGHT_Y_REVERSE = -1.0
    DEADZONE = 0.08


class JoystickXONE:
    # d-pad
    UP = 0  # UP
    DOWN = 1  # DOWN
    ROTATE_LEFT = 2  # LEFT
    ROTATE_RIGHT = 3  # RIGHT

    # bumper triggers
    TAKEOFF = 9  # RB
    LAND = 8  # LB

    # buttons
    FORWARD = 14  # Y
    BACKWARD = 11  # A
    LEFT = 13  # X
    RIGHT = 12  # B

    # axis
    LEFT_X = 0
    LEFT_Y = 1
    RIGHT_X = 2
    RIGHT_Y = 3
    LEFT_X_REVERSE = 1.0
    LEFT_Y_REVERSE = -1.0
    RIGHT_X_REVERSE = 1.0
    RIGHT_Y_REVERSE = -1.0
    DEADZONE = 0.09


prev_flight_data = None
video_player = None

def update_drone_img():
    pyautogui.press('s')
    path = os.getcwd()
    filelist = os.listdir(path)
    for image in filelist:
        if image.endswith(".png"):
            return image
        else:
            non_img = 'img.png'
            return non_img


def check_drone_img():
    pyautogui.press('s')
    path = os.getcwd()
    filelist = os.listdir(path)
    for image in filelist:
        if image.endswith(".png"):
            read = cv2.imread("img.png")
            cv2.imshow("captured", read)
            cv2.waitKey(1)


def handler(event, sender, data, **args):
    global prev_flight_data
    global video_player

    drone = sender

    if event is drone.EVENT_FLIGHT_DATA:
        if prev_flight_data != str(data):
            prev_flight_data = str(data)
    elif event is drone.EVENT_VIDEO_FRAME:
        if video_player is None:
            video_player = Popen(['mplayer', '-fps', '35', '-'], stdin=PIPE)
        try:
            video_player.stdin.write(data)

        except IOError as err:
            print(err)
            video_player = None
    else:
        print('event="%s" data=%s' % (event.getname(), str(data)))


def update(old, new, max_delta=0.3):
    if abs(old - new) <= max_delta:
        res = new
    else:
        res = 0.0
    return res


def main():
    pygame.init()
    pygame.joystick.init()
    buttons = None
    try:
        js = pygame.joystick.Joystick(0)
        js.init()
        js_name = js.get_name()
        print('Joystick name: ' + js_name)

        if js_name in ('Wireless Controller', 'Sony Computer Entertainment Wireless Controller', 'PS4 Controller'):
            buttons = JoystickPS4
            print(buttons)
        elif js_name in ('PLAYSTATION(R)3 Controller', 'Sony PLAYSTATION(R)3 Controller'):
            buttons = JoystickPS3
        elif js_name == 'Xbox One Wired Controller':
            buttons = JoystickXONE
    except pygame.error:
        pass

    if buttons is None:
        print('no supported joystick found, please connect the controller')
        return

    drone = tellopy.Tello()
    drone.connect()
    drone.start_video()
    drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
    drone.subscribe(drone.EVENT_VIDEO_FRAME, handler)
    speed = 100

    try:
        while 1:
            # loop with pygame.event.get() is too much tight w/o some sleep
            time.sleep(0.01)
            for e in pygame.event.get():
                # print(e)
                if e.type == pygame.locals.JOYAXISMOTION:
                    continue

                elif e.type == pygame.locals.JOYHATMOTION:
                    if e.value[0] < 0:
                        drone.counter_clockwise(speed)
                    if e.value[0] == 0:
                        drone.clockwise(0)
                    if e.value[0] > 0:
                        drone.clockwise(speed)
                    if e.value[1] < 0:
                        drone.down(speed)
                    if e.value[1] == 0:
                        drone.up(0)
                    if e.value[1] > 0:
                        drone.up(speed)
                elif e.type == pygame.locals.JOYBUTTONDOWN:
                    if e.button == buttons.TAKEOFF:
                        drone.takeoff()
                        print('telll')
                        speak.Speak('Taking Off')

                        # tello.takeoff()
                    if e.button == buttons.LAND:
                        drone.land()
                        speak.Speak('Landing')

                    elif e.button == buttons.UP:
                        drone.up(speed)
                    elif e.button == buttons.DOWN:
                        drone.down(speed)
                    elif e.button == buttons.ROTATE_RIGHT:
                        drone.clockwise(speed)
                    elif e.button == buttons.ROTATE_LEFT:
                        drone.counter_clockwise(speed)
                    elif e.button == buttons.FORWARD:
                        drone.forward(speed)
                    elif e.button == buttons.BACKWARD:
                        drone.backward(speed)
                    elif e.button == buttons.RIGHT:
                        drone.right(speed)
                    elif e.button == buttons.LEFT:
                        drone.left(speed)
                    elif e.button == buttons.FLIPR:
                        drone.flip_right()
                    elif e.button == buttons.TAKEPHOTO:
                        pyautogui.press('s')
                elif e.type == pygame.locals.JOYBUTTONUP:

                    print('ps4----')
                    if e.button == buttons.TAKEOFF:
                        drone.takeoff()
                        speak.Speak('Taking Off')
                    elif e.button == buttons.UP:
                        drone.up(0)
                    elif e.button == buttons.DOWN:
                        drone.down(0)
                    elif e.button == buttons.ROTATE_RIGHT:
                        drone.clockwise(0)
                    elif e.button == buttons.ROTATE_LEFT:
                        drone.counter_clockwise(0)
                    elif e.button == buttons.FORWARD:
                        drone.forward(0)
                    elif e.button == buttons.BACKWARD:
                        drone.backward(0)
                    elif e.button == buttons.RIGHT:
                        drone.right(0)
                    elif e.button == buttons.LEFT:
                        drone.left(0)
                    elif e.button == buttons.FLIPR:
                        drone.flip_right(0)
                    elif e.button == buttons.TAKEPHOTO:
                        speak.Speak('flying mode')
                        pyautogui.press('s')

    except KeyboardInterrupt as e:
        print(e)
    except Exception as e:
        print(e)

    drone.quit()
    exit(1)


def detect_hands(hand_image, hands_model, draw_hands=0):
    result_hands = [0] * 4

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    hand_dict = {0: 'one', 1: 'three', 2: 'two'}

    model = hands_model

    with mp_hands.Hands(
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while True:  # while cam is accessible
            image = cv2.imread(hand_image)
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
                image = hand_image
                image = cv2.imread(image)
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
    # cap.release()
    return result_hands


def show_hands_result():
    with open('./logistic2.pkl', 'rb') as f:
        my_model = pickle.load(f)
    hands_info = [0] * 4
    count = 0
    while True:
        pyautogui.press('s')
        path = os.getcwd()
        filelist = os.listdir(path)
        print("before for loop")

        for image in filelist:  # filelist[:] makes a copy of filelist.
            if image.endswith(".png"):
                if count is 0:
                    speak.Speak('安全なら1、救助が必要な場合は2、緊急な救助が必要な場合は3と指で示してください')
                    count = count + 1
                hands_info = detect_hands(image, my_model, 0)
                print(hands_info)
                if hands_info[2] is 'one':
                    speak.Speak('一と検知しました')
                elif hands_info[2] is 'two':
                    speak.Speak('二と検知しました')
                elif hands_info[2] is 'three':
                    speak.Speak('3と検知しました')
                os.remove(image)
            if image is None:
                cv2.destroyAllWindows()


def detect_person_hand():
    yolo_v7()
    show_hands_result()


if __name__ == '__main__':
    # initialization for yolo v7
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()

    # parallel processing
    # process_display = threading.Thread(target=yolo_v7)
    # process_display = threading.Thread(target=check_drone_img)  # for debug
    process_display = threading.Thread(target=detect_person_hand)
    process_joystick = threading.Thread(target=main)

    process_display.start()
    process_joystick.start()

    process_display.join()
    process_joystick.join()
