import numpy as np
import cv2
from matplotlib import pyplot as plt

wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

roi=0
while 1:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0), 2)

    cv2.imshow('img',frame)
    
    if cv2.waitKey(1) == ord('q'): #q를 누르면 반복문 종료
        break
cam.release()  #카메라 사용 종료
cv2.destroyAllWindows()  #이미지 창 종료
