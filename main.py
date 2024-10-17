import cv2 
import numpy as np
import track
import tensorflow
from tensorflow import keras

tracker = track.HandsTrack()
cam_ = cv2.VideoCapture(0)
model_a = keras.models.load_model('model_alpha.h5')

ans = 'abcdefghijklmnopqrstuvwxyz'

def inModel(img):
    return np.array(img).reshape(-1,28*28)
while True:
    success, img = cam_.read()
    if (not success):
        print("Capture not found!")
        break
    cv2.imshow("Hand Detector", tracker.tracking_only_hands(img)) 
    predict = model_a.predict(inModel(tracker.tracking_only_hands(img)))
    print(predict)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break