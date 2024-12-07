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
                x_coords = [landmark.x for landmark in landmark_iter.landmark]
                y_coords = [landmark.y for landmark in landmark_iter.landmark]

                black_image = np.zeros_like(img)

                for landmark in landmark_iter.landmark:
                    h_, w_, _ = black_image.shape
                    x = int(landmark.x * w_) 
                    y = int(landmark.y * h_) 
                    cv2.circle(black_image, (x, y), 5, (255, 255, 255), -1)

                # Определяем границы квадрата
                x_min = int(min(x_coords) * img.shape[1]) - crop_size
                x_max = int(max(x_coords) * img.shape[1]) + crop_size
                y_min = int(min(y_coords) * img.shape[0]) - crop_size
                y_max = int(max(y_coords) * img.shape[0]) + crop_size
                
                black_image = black_image[y_min : y_max,x_min : x_max]
                try:
                    black_image = cv2.resize(black_image,(32,32), interpolation=cv2.INTER_NEAREST)
                except:
                    self.isEmpty = True
                    return img
                black_image = cv2.cvtColor(black_image[: ,::-1],cv2.COLOR_RGB2GRAY)
                self.isEmpty = False
                return black_image
        self.isEmpty = True
        return img
    
    

        





