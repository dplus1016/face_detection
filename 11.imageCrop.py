import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

img_name=input("성명(영문 이니셜): ")
img_num=int(input("이미지 개수: "))
crpoed_img_num=1
# For static images:
with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.1) as face_detection:
  for i in range(img_num):
    image = cv2.imread(f"./images/{img_name}/{img_name}{i+1}.jpg")
    if image is None: continue
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw face detections of each face.
    if not results.detections: continue
    annotated_image = image.copy()
    for detection in results.detections:
      rsp, rep=mp_drawing.draw_detection(annotated_image, detection)
      if rsp==None or rep==None: continue
      if rsp[0]==rep[0] or rsp[1]==rep[1]: continue
      annotated_image=annotated_image[rsp[1]:rep[1], rsp[0]:rep[0]]
      print(f"rsp[1]: {rsp[1]}, rep[1]: {rep[1]}, rsp[0]: {rsp[0]}, rep[0]: {rep[0]}")
      # cv2.imshow('img',annotated_image)
      print(i+1)
      cv2.imshow('croped', annotated_image)
      #cv2.waitKey(0)
      cv2.imwrite(f'./croped/{img_name}/{img_name}{crpoed_img_num}_croped.png', annotated_image)
      crpoed_img_num+=1

