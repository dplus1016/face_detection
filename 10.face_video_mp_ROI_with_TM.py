import cv2
from keras.models import load_model
import numpy as np
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.VideoCapture(0)
#model = load_model('model/glasses.h5')
model = load_model('model/persons.h5')

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
    f=open("model/persons.txt","r",encoding="utf8")
    label=f.readlines()    
    return label[i][2:-1]

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  
  prefaces=0  
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)
    
    if results.detections!=None:
        roi=[]*len(results.detections)
        roi_cnt=0
        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
            # ADH 수정_ROIs 추출
                #if not mp_drawing.draw_detection(image, detection): break
                rsp, rep=mp_drawing.draw_detection(image, detection)
                if rsp==None or rep==None: break
                roi.append(image[rsp[1]:rep[1] , rsp[0]:rep[0]])
                roi_cnt+=1

            # ADH 수정_ROIs 창 생성
        for i in range(roi_cnt):
            roi[i]=cv2.resize(roi[i],(200,200))

            preprocessed = preprocessing(roi[i])
            prediction = predict(preprocessed)
            font=cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image,prediction,(rsp[0],rsp[1]-10),font,1,(0,255,0),2,cv2.LINE_AA)
            
            cv2.imshow('face'+str(i+1), roi[i])

        # ADH 수정_
        if prefaces>roi_cnt:
            for i in range(prefaces,roi_cnt,-1):
                cv2.destroyWindow('face'+str(i))
    
        del roi
        prefaces=roi_cnt

        # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Detection', image)
        
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

# For static images:
'''
IMAGE_FILES = []
with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) as face_detection:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw face detections of each face.
    if not results.detections:
      continue
    annotated_image = image.copy()
    for detection in results.detections:
      print('Nose tip:')
      print(mp_face_detection.get_key_point(
          detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
      mp_drawing.draw_detection(annotated_image, detection)
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
'''
