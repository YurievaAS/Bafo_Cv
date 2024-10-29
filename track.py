import cv2 
import numpy as np
import mediapipe as mp
import time

class HandsTrack:
    def __init__(self,mode=False, maxHands = 2, model_complexity=1,min_detection_confidence=0.5, min_tracking_confidence=0.6):
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mode = mode
        self.maxHands = maxHands
        
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(self.mode, self.maxHands,self.model_complexity, self.min_detection_confidence, self.min_tracking_confidence)
    
        self.isEmpty = True

    def tracking(self, img):
        h, w = img.shape[0:2]
        x, y = 0,0
        crop_size = 20
        self.results = self.hands.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        if self.results.multi_hand_landmarks:
            for landmark_iter in self.results.multi_hand_landmarks:
                
                x_min = min(int(landmark.x * img.shape[1]) for landmark in landmark_iter.landmark) - crop_size
                x_max = max(int(landmark.x * img.shape[1]) for landmark in landmark_iter.landmark) + crop_size
                y_min = min(int(landmark.y * img.shape[0]) for landmark in landmark_iter.landmark) - crop_size
                y_max = max(int(landmark.y * img.shape[0]) for landmark in landmark_iter.landmark) + crop_size
                
                # _min_ = min(y_min,x_min)
                # _max_ = max(y_max,x_max)
                # img = img[_min_:_max_,_min_:_max_]
                img = img[y_min:y_max,x_min:x_max]
                img = cv2.resize(img,(28,28), interpolation=cv2.INTER_LINEAR)
                img = cv2.cvtColor(img[: ,::-1],cv2.COLOR_RGB2GRAY)
                self.isEmpty = False
                return img
        self.isEmpty = True
        return img
    
        





