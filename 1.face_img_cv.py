import numpy as np
import cv2
from matplotlib import pyplot as plt

classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
img = cv2.imread('img2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = classifier.detectMultiScale(gray)

for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0), 2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()