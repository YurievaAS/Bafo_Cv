import hand_detector_class as hdc
import cv2
import numpy as np
import matplotlib as mlp
import os
import time
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import mediapipe as mp

detector = hdc.HandDetector()
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if (not success):
        break
    detector.find_hands(img)
    print(0, detector.hand_position(img)[0:5])
    print(1, detector.hand_position(img, 1)[0:5]) #показывает 1-е 5 элементов, чтобы загружено не было

