import cv2
import numpy as np
import matplotlib as mlp
import os
import time
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import mediapipe as mp
cap = cv2.VideoCapture(0) #считывание данных с камеры
#cap.set(3, 300)
#cap.set(4, 500)
mpHands = mp.solutions.hands #доступ к функциональности, связанной с обнаружением рук
hands = mpHands.Hands() #создание объекта на основе класса
mpDraw = mp.solutions.drawing_utils #доступ к функциональности

pTime = 0
cTime = 0
while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    #results.multi_hand_landmarks - найдены руки или нет
    if (results.multi_hand_landmarks != None):
        for hand_lm in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_lm.landmark):
                height, width, channels = img.shape
                x, y = int(lm.x*width), int(lm.y*height) #преобразование landmarks в координаты
                print(id," :",  x, y)
            mpDraw.draw_landmarks(img, hand_lm, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_DUPLEX,3,(125,67,234),10 )

    cv2.imshow('Video', img)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
