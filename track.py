import cv2 
import numpy as np
import mediapipe as mp

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
                crop_size = 600
                x = int(land_mark.landmark[self.mp_hands.HandLandmark.WRIST].x * w)
                y = int(land_mark.landmark[self.mp_hands.HandLandmark.WRIST].y * h)
        
                x_min = max(1, x - crop_size)
                x_max = min(w, x + crop_size)
                y_min = max(1, y - crop_size)
                y_max = min(h, y)

                img = img[y_min:y_max, x_min:x_max]

            return cv2.cvtColor(cv2.resize(img,(28,28), interpolation=cv2.INTER_LINEAR),cv2.COLOR_RGB2GRAY)

        return cv2.cvtColor(img[0:28,0:28][: ,::-1],cv2.COLOR_RGB2GRAY)

        



