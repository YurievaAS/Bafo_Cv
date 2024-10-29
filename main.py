import cv2 
import numpy as np
import track as track
import tensorflow
from tensorflow import keras
import time

tracker = track.HandsTrack()
cam_ = cv2.VideoCapture(0)

empty_img = np.zeros(10)

success,img = cam_.read()

model = keras.models.load_model('models/model.keras')

def to_numpy_array(img):
    return np.array(img).reshape(1,28,28,1)
prediction = None

while True:
    success, img = cam_.read()
    if (not success):
        print("Capture not found!")
        break
    try:
        tracker.isEmpty = False
        img = tracker.tracking(img)
        cv2.imshow('Hand tracker', img)
    except:
        tracker.isEmpty = True
        cv2.imshow('Hand tracker', empty_img)

    if tracker.isEmpty == False:
        prediction = model.predict(to_numpy_array(img)) 
        print(prediction)
        
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
    time.sleep(0.1)