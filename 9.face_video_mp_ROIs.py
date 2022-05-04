import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.VideoCapture(0)
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
            cv2.imshow('face'+str(i+1), roi[i])

        # ADH 수정_
        if prefaces>roi_cnt:
            for i in range(prefaces,roi_cnt,-1):
                cv2.destroyWindow('face'+str(i))
    
        del roi
        prefaces=roi_cnt

        # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
        
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