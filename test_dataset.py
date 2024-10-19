import cv2
import hand_detector_class as hdc
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
def f(x):
    if (x <= 0):
        return 0
    elif (x >= 500):
        return 500
    else:
        return x
decrypt = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           'hello', 'thank you', 'please', 'yes', 'no', 'sorry', 'help', 'more',
           'love', 'friend', 'family', 'sorry', 'me', 'mine', 'her/ his', 'she/ he',
           'food', 'water', 'good', 'bad', 'happy', 'sad', 'like', 'want', 'need',
           'time', 'where', 'who', 'what', 'why', 'how', 'have', 'come', 'go', 'stop',
           'wait', 'see', 'look', 'listen', 'talk', 'think', 'understand', 'play',
           'work', 'phone', 'do', 'dont', 'tired', 'hungry', 'thirsty', 'okay',
           'interesting', 'funny', 'be careful', 'take care', 'welcome', 'beautiful',
           'amazing', 'awesome', 'great job', 'busy', 'free', 'its', 'excited',
           'question', 'agree', 'disagree', 'cold', 'hot']
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands #доступ к функциональности
hands = mpHands.Hands(min_tracking_confidence=0.9, min_detection_confidence=0.2)

for label in range(1):
    print("collecting images for no")
    cnt = 0
    while cnt < 1000:
        success, img = cap.read()
        cv2.resize(img,(500,500))
        if (not success):
            break
        #qcv2.imshow("data", img)
        height, width, channels = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        var = 250
        if results.multi_hand_landmarks: #если руки найдены
            for hand_landmarks in results.multi_hand_landmarks: #координаты для рисования квадрата
                x = int(hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].x * width)
                y = int(hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].y * height)
                print(x,y)
                img_save = 'C:/Users/arish/PycharmProjects/Bafo_Cv/img_for_train/no/'+ 'no' + str(cnt)+ '.png'
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                cv2.imwrite(img_save, img[f(x - var):f(x + var) ,f(y- var):f(y + var)])
                print('written!\n')
                print(cnt)
                cnt += 1
                time.sleep(0.2)
    time.sleep(10)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break


""""
0 2252
1 2020
2 2288
3 2392
4 1914
5 2408
6 2180
7 2026
8 2324
9 0
10 2228
11 2482
12 2110
13 2302
14 2392
15 2176
16 2558
17 2588
18 2398
19 2372
20 2322
21 2164
22 2450
23 2328
24 2236
25 0 
!Нет букв J и Z, тк они динамические, надо придумать решение
"""