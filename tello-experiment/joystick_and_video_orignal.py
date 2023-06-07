import time
import os
import sys
import tellopy
import pygame
import pygame.locals
from subprocess import Popen, PIPE
import cv2
import os
import pyautogui
import win32com.client as wincl
speak = wincl.Dispatch("SAPI.SpVoice")
import threading
import tello


class JoystickPS3:
    # d-pad
    UP = 4  # UP
    DOWN = 6  # DOWN
    ROTATE_LEFT = 7  # LEFT
    ROTATE_RIGHT = 5  # RIGHT

    # bumper triggers
    TAKEOFF = 11  # R1
    LAND = 10  # L1
    # UNUSED = 9 #R2
    # UNUSED = 8 #L2

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
    TAKEPHOTO = 7 #R2
    FLIPR = 6 #L2

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
    # UNUSED = 7 #RT
    # UNUSED = 6 #LT

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
                #print('press s')

        except IOError as err:
            print("no video player")
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

def img_display():
    CONFIDENCE_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.4
    COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
    class_names = []
    with open("classes.txt", "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]
    net = cv2.dnn.readNet("yolov3-tiny_face_best.weights", "yolov3-tiny_face.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    #net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)
    while(1):
        pyautogui.press('s')
        path = os.getcwd()
        filelist = os.listdir(path)
        for fichier in filelist:  # filelist[:] makes a copy of filelist.
            if (fichier.endswith(".png")):
                #filelist.remove(fichier)
                frame = cv2.imread(fichier)
                # cv2.imshow('Fra', frame)
                # cv2.waitKey(1)
                start = time.time()
                print(CONFIDENCE_THRESHOLD)
                print(NMS_THRESHOLD)
                classes, scores, boxes = model.detect(frame, 0.4, NMS_THRESHOLD)
                end = time.time()

                start_drawing = time.time()
                for (classid, score, box) in zip(classes, scores, boxes):
                    #color = COLORS[int(classid) % len(COLORS)]
                    color = (255, 255, 0)
                    label = "%s : %f" % (class_names[0], score)
                    cv2.rectangle(frame, box, color, 2)
                    cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                end_drawing = time.time()

                # fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (
                # 1 / (end - start), (end_drawing - start_drawing) * 1000)
                fps_label = "FPS: %.2f " % (1 / (end - start))
                cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                cv2.imshow('YOLO-Tiny3', frame)
                cv2.waitKey(1)
                #cv2.destroyAllWindows()
                os.remove(fichier)
            if fichier is None:
                cv2.destroyAllWindows()

        #print(filelist)



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
                #print(e)
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

                        #tello.takeoff()
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
                        #speak.Speak('Training Mode')
                        # print('training mode')
                        # #drone.right(speed)
                        # with open(r"C:\Users\litao\Desktop\tao_tello_face\src\mode.txt", "w") as text_file:
                        #     print(f"tm", file=text_file)

                    elif e.button == buttons.LEFT:
                        drone.left(speed)
                        # speak.Speak('Recognition Mode')
                        # print('recognition mode')
                        # with open(r"C:\Users\litao\Desktop\tao_tello_face\src\mode.txt", "w") as text_file:
                        #     print(f"rm", file=text_file)
                        # #drone.left(speed)
                    elif e.button == buttons.FLIPR:
                        drone.flip_right()
                        #pyautogui.press('enter')

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
                        #pyautogui.press('enter')
                    elif e.button == buttons.TAKEPHOTO:
                        speak.Speak('flying mode')
                        pyautogui.press('s')


    except KeyboardInterrupt as e:
        print(e)
    except Exception as e:
        print(e)

    drone.quit()
    exit(1)


if __name__ == '__main__':
    #main()
    # frame = cv2.imread('shot0002.png')
    # print(frame)
    # cv2.imshow('img', frame)
    # cv2.waitKey(1)
    #img_display()
    t1 = threading.Thread(target=main)
    t2 = threading.Thread(target=img_display)
    t1.start()
    t2.start()
    t1.join()
    t2.join()