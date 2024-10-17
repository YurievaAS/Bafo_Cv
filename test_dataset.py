import cv2
import hand_detector_class as hdc
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
"""
train = pd.read_csv("sign_mnist_train.csv")
test = pd.read_csv("sign_mnist_train.csv")

tr = train.iloc[:, 0]
ts = test.iloc[:, 0]
"""

def uns(x):
    if x <= 0:
        return 0
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
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
number_of_ims = 10
#for label in range(26,33):
while True:
    #time.sleep(10)
    #print("collecting images for " + decrypt[label])
    for cur_img in range(number_of_ims):
        success, img = cap.read()
        if (not success):
            break
        cv2.imshow("data", img)
        height, width, channels = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        x = []
        y = []
        if results.multi_hand_landmarks: #если руки найдены
            for hand_landmarks in results.multi_hand_landmarks: #координаты для рисования квадрата
                x.append(int(hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].x * width))
                y.append(int(hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].y* height))
                print(x, y)
            ind = x.index(max(x))
            ind1 = x.index(min(x))
            div_x = max(x) - min(x)
            div_y = max(y) - min(y)

            var = 300
            print(x, y)
            #cv2.rectangle(img, , (193, 182, 255), 2)
        cv2.imshow("data", img)
        #time.sleep(10)
        #img_save = 'C:/Users/arish/PycharmProjects/Bafo_Cv/img_for_train/'+ 'test_' + decrypt[label] + str(cur_img)+ '.png'
        #cv2.imwrite(img_save, img)
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