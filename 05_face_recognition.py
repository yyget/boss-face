''''
Real Time Face Recogition
	==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc                       
	==> LBPH computed model (trained faces) should be on trainer/ dir
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18  

'''
# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import os
import pymouse,pykeyboard,sys
import time
from pymouse import PyMouse
from pykeyboard import PyKeyboard

m = PyMouse()
k = PyKeyboard()
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Yang', 'Jing', 'Peng', 'MaDong', 'W'] 

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
isHide = 0
while True:
    hide = 0
    ret, img =cam.read()
    #img = cv2.flip(img, -1) # Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            if id == 'MaDong':
                #isHide = 1
                hide = 1
                #if id == 'Yang':
                    #isHide == 1
                    #k.press_keys([k.windows_l_key,'d'])
                    #k.press_key(k.alt_key)
                    #k.tap_key(k.tab_key)
                    #k.release_key(k.alt_key)
                    #break
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    if hide == 1 and isHide == 0:
        print("hide")
        isHide = 1
        
        #k.press_keys([k.windows_l_key,'m'])
        #k.press_keys([k.windows_l_key,'d'])
        k.press_keys([k.windows_l_key,k.control_key,k.right_key])
        #time.sleep(1)
            #k.press_key(k.alt_key)
            #k.tap_key(k.tab_key)
            #k.release_key(k.alt_key)
            #k.press_keys([k.windows_l_key,'d'])
            #k.release_keys([k.windows_l_key,'d'])
            
    elif hide == 0 and isHide == 1:
        print("show")
        #time.sleep(1)
        isHide = 0
        #k.press_keys([k.windows_l_key,k.shift_key,'m'])
        #k.press_keys([k.windows_l_key,'d'])
        k.press_keys([k.windows_l_key,k.control_key,k.left_key])
            #print("show")
            #k.press_key(k.alt_key)
            #k.tap_key(k.tab_key)
            #k.release_key(k.alt_key)
            #k.press_keys([k.windows_l_key,'d'])
            #k.release_keys([k.windows_l_key,'d'])
            
    cv2.imshow('camera',img) 

    key = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if key == 27:
        break
    time.sleep(0.2)

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
