import cv2
import numpy as np
import matplotlib as mlp
import os
import time
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import mediapipe as mp

class HandDetector():
    #конструктор класса (все сложности можно убрать)
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.4, trackCon=0.6):
        self.pTime = 0
        #параметры модели
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands #доступ к функциональности
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity,\
                                        self.detectionCon, self.trackCon) # создание объекта
        self.mpDraw = mp.solutions.drawing_utils #доступ к функциональности

    #метод класса, отвечающий за отображение
    def find_hands(self,img, draw=True, fps=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        if (results.multi_hand_landmarks):
            for hand_lm in results.multi_hand_landmarks:
                if (draw):
                    self.mpDraw.draw_landmarks(img, hand_lm, self.mpHands.HAND_CONNECTIONS)
        if (fps):
            cTime = time.time()
            fps = 1 / (cTime - self.pTime)
            self.pTime = cTime
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (125, 67, 234), 10)
        return img

    #метод класса, вовзращающий за коор-ты точек (id на руке - номер в массиве)
    def hand_position(self, img):
        lmList = []
        box = []
        bbox1 = [0, 0,0,0]
        bbox2 = [0, 0,0,0]
        height, width, channels = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        elsements = self.results.multi_hand_landmarks
        if (elsements):
            for hand_num in range(len(elsements)):
                cx= 0
                cy = 0
                xList = []
                yList = []
                myHand = elsements[hand_num]
                for id, lm in enumerate(myHand.landmark):
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    xList.append(cx)
                    yList.append(cy)
                    lmList.append([cx, cy])

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                if (hand_num == 0):
                    bbox1 = xmin, ymin, xmax, ymax
                if (hand_num == 1):
                    bbox2 = xmin, ymin, xmax, ymax
                xList = []
                yList = []
            #if (len(elsements) == 1):
            cv2.rectangle(img, (bbox1[0] - 20, bbox1[1] - 20), (bbox1[2] + 20, bbox1[3] + 20), (0, 255, 0), 2)
            if (len(elsements) == 2):
                cv2.rectangle(img, (bbox2[0] - 20, bbox2[1] - 20), (bbox2[2] + 20, bbox2[3] + 20), (0, 255, 0), 2)
        return img, lmList


