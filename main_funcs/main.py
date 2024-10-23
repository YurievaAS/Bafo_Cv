from main_funcs import hand_detector_class as hdc
import cv2
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

detector = hdc.HandDetector()
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if (not success):
        break
    detector.find_hands(img)
    cv2.imshow("Hand Detector", detector.hand_position(img)[0])
    #print(0, detector.hand_position(img)[1][0:5])
    #print(1, detector.hand_position(img, 1)[1][0:5]) #показывает 1-е 5 элементов, чтобы загружено не было
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
