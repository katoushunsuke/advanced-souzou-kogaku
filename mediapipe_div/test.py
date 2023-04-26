import numpy as np
import cv2
import mediapipe as mp
import random
import pickle

label_dict = {0: 'b', 1: 'a', 2: 'c'}
dic = {"a": "one", "b": "two", "c": "three"}

with open('./model/logistic.pkl', 'rb') as f:
    model = pickle.load(f)

hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()