import cv2
import os
import time
import keyboard
decrypt = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j_0', 'j_1', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z_0','z_1',
           'hello', 'yes_0','yes_1','no','i_me','help','she_he','please','thank you' , 'sorry']
cap = cv2.VideoCapture(0)
print("collecting images for j_letter")
cnt = 0
while cnt < 1000:
    success, img = cap.read()
    if (not success):
        break
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(64, 64))
    cv2.imshow('data', img)
    if keyboard.is_pressed("enter"):
        img_save = 'C:/Users/arish/PycharmProjects/Bafo_Cv/img_for_train/j_letter/j_letter_1/'+ 'j_letter_1' \
                                                                                                '' \
                                                                                                '_' + str(cnt)+ '.png'
        cv2.imwrite(img_save, img)
        print(cnt, 'written!\n')
        cnt += 1
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break





""""
0 2252
1 2020
2 2288
3 2392
4 1914
5 2408
6 2180
7 2026
8 2324
9 0
10 2228
11 2482
12 2110
13 2302
14 2392
15 2176
16 2558
17 2588
18 2398
19 2372
20 2322
21 2164
22 2450
23 2328
24 2236
25 0 
!Нет букв J и Z, тк они динамические, надо придумать решение
"""