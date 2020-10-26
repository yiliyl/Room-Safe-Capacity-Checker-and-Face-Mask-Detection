import time
import cv2
import numpy as np
from mask_detection import faceNet,maskNet,detect_and_predict_mask
confid = 0.5
thresh = 0.5

vid_path = "4.mp4"
save_path="o4.mp4"
room_safe_capacity = 3

labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)

weightsPath = "yolov3.weights"
configPath = "yolov3.cfg"

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

vs = cv2.VideoCapture(vid_path)
writer = None
(W, H) = (None, None)

q = 0
while True:
    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]
        q = W

    frame = frame[0:H, 0:q]
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if LABELS[classID] == "person":

                if confidence > confid:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)

    if len(idxs) > 0:

        status = list()
        idf = idxs.flatten()
        center = list()
        for i in idf:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            center.append([int(x + w / 2), int(y + h / 2)])

            status.append(0)

        total_p = len(center)
        safe_p = status.count(0)
        kk = 0
        for i in idf:

            cv2.putText(frame, "Confernce Room GPTW wrt. COVID-19", (50, 45),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
           
            tot_str = "TOTAL COUNT: " + str(total_p)

            safe_str = "SAFE COUNT: " + str(room_safe_capacity)

            if int(total_p)>room_safe_capacity:
                total_color=(0, 0, 150)
            else:
                total_color=(255, 255, 255)
            cv2.putText(frame, tot_str, (10, H - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, total_color, 2)
            cv2.putText(frame, safe_str, (10, H - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            #------------detect wearing a  mask or not--------------
            f=frame[y:y+h,x:x+w]
            (locs, preds) = detect_and_predict_mask(f, faceNet, maskNet)
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                if withoutMask > mask:
                    color = (0,0,255)
                    label = "{}: {:.2f}%".format("No Mask", max(mask, withoutMask) * 100)
                else:
                    color = (0, 255, 0)
                    label = "{}: {:.2f}%".format("Mask", max(mask, withoutMask) * 100)

                cv2.putText(frame, label, (startX+x, startY+y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                cv2.rectangle(frame, (startX+x, startY+y), (endX+x, endY+y), color, 2)
            
            if status[kk] == 1:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)
            elif status[kk] == 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 120, 255), 2)
            kk += 1

        cv2.imshow('Conference Room GPTW wrt. Covid19', frame)
        cv2.waitKey(1)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX') #MJPG/MP4V
        writer = cv2.VideoWriter(save_path, fourcc, 30,(frame.shape[1], frame.shape[0]), True)
    writer.write(frame)

print("Processing finished: open {}".format(save_path))
writer.release()
vs.release()


