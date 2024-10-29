import cv2 
import numpy as np
import track
from tensorflow import keras
import time
from PyQt5.QtWidgets import QApplication
import sys
import ui
words = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
           'hello', 'yes','no','i_me','help','she_he']
def to_numpy_array(img):
    return np.array(img).reshape(1,28,28,1)

app = QApplication(sys.argv)
window = ui.MainWindow()

tracker = track.HandsTrack()
cam_ = cv2.VideoCapture(0)

model = keras.models.load_model('models/model.keras')
prediction = None

while True:
    success, img = cam_.read()
    window.update_frame(img)
    window.show()
    if (not success):
        print("Capture not found!")
        break
    img = tracker.tracking(img)

    if tracker.isEmpty == False:
        prediction = model.predict(to_numpy_array(img))
        predicted_word = np.argmax(prediction)
        print(words[predicted_word])

        
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
    time.sleep(0.1)