from keras.models import load_model
import numpy as np
import cv2
from matplotlib import pyplot as plt

wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
#model = load_model('model/glasses.h5')
model = load_model('model/persons_v4.h5')

# 이미지 처리하기
def preprocessing(frame):
    size= (224, 224)  # 사이즈 조정 티쳐블 머신에서 사용한 이미지 사이즈로 변경해준다.
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    frame_normalized= (frame_resized.astype(np.float32) / 127.0) - 1  # 이미지 정규화
    frame_reshaped=frame_normalized.reshape((1, 224, 224, 3))  # keras 모델에 전달할 올바른 모양의 배열 생성
    return frame_reshaped

# 예측용 함수
def predict(frame):
    prediction = model.predict(frame)
    p=prediction[0]
    for i in range(len(p)): # i = 0~2
        if p[i] == max(p):
            break
    #f=open("model/glasses.txt","r",encoding="utf8")
    f=open("model/persons_v4.txt","r",encoding="utf8")
    label=f.readlines()    
    return label[i][2:-1]

while 1:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0), 2)
        roi=frame[y:y+h, x:x+w]

    preprocessed = preprocessing(roi)
    prediction = predict(preprocessed)
    print(prediction)
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,prediction,(x,y-10),font,1,(0,255,0),2)

    roi=cv2.resize(roi, (200,200))
    cv2.imshow('img',frame)
    cv2.imshow('roi',roi)

    if cv2.waitKey(1) == ord('q'): #q를 누르면 반복문 종료
        break
cam.release()  #카메라 사용 종료
cv2.destroyAllWindows()  #이미지 창 종료
