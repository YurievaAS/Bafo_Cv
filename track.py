import cv2 
import numpy as np
import mediapipe as mp
import time

class HandsTrack:
    def __init__(self,mode=False, maxHands = 2, model_complexity=1,min_detection_confidence=0.4, min_tracking_confidence=0.6):
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mode = mode
        self.maxHands = maxHands

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(self.mode, self.maxHands,self.model_complexity, self.min_detection_confidence, self.min_tracking_confidence)
   
    def tracking(self, img):
        self.results = self.hands.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        if self.results.multi_hand_landmarks:
            for hand_iter in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_iter ,self.mp_hands.HAND_CONNECTIONS)

        return img[ : , ::-1]
    
    def tracking_only_hands(self,img):
        h, w = img.shape[0:2]
        x, y = 0,0
        self.results = self.hands.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        if self.results.multi_hand_landmarks:
            for land_mark in self.results.multi_hand_landmarks:
                
                x = int(land_mark.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * w)
                y = int(land_mark.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * h)
                crop_size = 230
                x_min = max(1, x - crop_size)
                x_max = min(w, x + crop_size)
                y_min = max(1, y - crop_size)
                y_max = min(h, y + crop_size)

                img = img[y_min:y_max, x_min:x_max]

            return cv2.cvtColor(cv2.resize(img,(64,64), interpolation=cv2.INTER_LINEAR),cv2.COLOR_RGB2GRAY) if img.size != 0 else img

        return cv2.cvtColor(img[: ,::-1],cv2.COLOR_RGB2GRAY)

    def modif_pict(self,img):
        return cv2.cvtColor(cv2.resize(img[100:700, 250:850],(64,64), interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY) if img.size != 0 else img

    def rectHands(self,img):
        h,w,_=img.shape
        self.results = self.hands.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        if self.results.multi_hand_landmarks:
            for hand_iter in self.results.multi_hand_landmarks:
                x = int(hand_iter.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * w)
                y = int(hand_iter.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * h)
                z = abs(int(hand_iter.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z*50))
                crop = z * 170
                img = cv2.rectangle(img,(x-crop,y-crop), (x+crop,y+crop),(255,0,0),3)

        return img[ : , ::-1]




