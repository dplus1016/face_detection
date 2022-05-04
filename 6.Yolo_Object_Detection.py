import cv2
import numpy as np

cam = cv2.VideoCapture(0)

classes = []
with open("yolo.names", "r") as f:
    for line in f.readlines():
        classes.append(line.strip())

# YOLO NETWORK 준비
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
names = net.getLayerNames()
layers = [names[i-1] for i in net.getUnconnectedOutLayers()]
print(layers)

while True:
    ret, frame = cam.read()
    h, w, c = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 
                                scalefactor = 1/255, 
                                size = (416, 416), 
                                mean = (0, 0, 0),
                                swapRB = True,
                                crop=False)
    net.setInput(blob)
    outs = net.forward(layers)
    class_ids = []
    confidences = []
    boxes = []
    h, w, c = frame.shape

    for out in outs:

        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:

                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                dw = int(detection[2] * w)
                dh = int(detection[3] * h)

                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes.append([x, y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            scores = confidences[i]
            cv2.rectangle(frame, 
                        (x, y), 
                        (x + w, y + h), 
                        (0, 0, 255), 5)
            cv2.putText(frame, 
                        label, 
                        (x, y - 20), 
                        cv2.FONT_ITALIC, 
                        0.5, 
                        (255, 255, 255), 
                        1)

    cv2.imshow("YOLOv3", frame)

    if cv2.waitKey(100) > 0: break
cam.release()
cv2.destroyAllWindows()

'''
['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote',
 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
'''
