import numpy as np
import cv2
from matplotlib import pyplot as plt

wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
prefaces=0

while 1:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray)
    
    roi=[]*len(faces)
    i=0
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0), 2)
        roi.append(frame[y:y+h, x:x+w])
        i+=1

    cv2.imshow('img',frame)
    for i in range(len(faces)):
        roi[i]=cv2.resize(roi[i],(200,200))
        cv2.imshow('face'+str(i+1), roi[i])
        
    if prefaces>len(faces):
        for i in range(prefaces,len(faces),-1):
            cv2.destroyWindow('face'+str(i))
    
    del roi
    prefaces=len(faces)

    if cv2.waitKey(1) == ord('q'): #q를 누르면 반복문 종료
        break
cam.release()  #카메라 사용 종료
cv2.destroyAllWindows()  #이미지 창 종료


'''
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0), 2)
        roi=frame[y:y+h, x:x+w]

    cv2.imshow('img',frame)
    cv2.imshow('roi',roi)


'''
