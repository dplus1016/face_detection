import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
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

    
    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    #if results.detections==None: continue
    if results.detections:
        for detection in results.detections:
    # ADH 수정_얼굴만 추출

    # [mediapipe 패키지 수정]
    # mediapipe 패키지 - python - solutions - drawing_utils.py - draw_detection 함수 끝에 아래의 내용 추가
    # return rect_start_point, rect_end_point

            rsp, rep=mp_drawing.draw_detection(image, detection)   
            if rsp==None or rep==None: break
            roi=image[rsp[1]:rep[1] , rsp[0]:rep[0]]

    # Flip the image horizontally for a selfie-view display.
    roi=cv2.resize(roi, (200,200))
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    cv2.imshow('roi',cv2.flip(roi,1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
