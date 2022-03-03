import numpy as np
import cv2
import os
from collections import Counter, defaultdict
import math
from yolo.detector import Detector

yolo_directory = 'yolo'
detection_confidence = 0.3
nms_threshold = 0.1
detector = Detector(yolo_directory, detection_confidence, nms_threshold)


idxGlobal = 0
# path of first frame
firstframe_path ='FrameNo0.png'

# Mobilenet files
prototxt = "mobilenet_data/MobileNetSSD_deploy.prototxt.txt"
model = "mobilenet_data/MobileNetSSD_deploy.caffemodel"

min_confidence = 0.3 # below this will be ignored

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLOR = (0, 0, 255)

# load our model
net = cv2.dnn.readNetFromCaffe(prototxt, model)

firstframe = cv2.imread(firstframe_path)
firstframe_gray = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
firstframe_blur = cv2.GaussianBlur(firstframe_gray,(21,21),0)

# path of video
file_path =r'video1.avi'

cap = cv2.VideoCapture(file_path)
consecutiveframe=20
track_temp=[]
track_master=[]
track_temp2=[]

top_contour_dict = defaultdict(int)
obj_detected_dict = defaultdict(int)

labelsPath = os.path.sep.join([yolo_directory, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

allObjects = []
def dist(p1, p2):     
    return (math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])) + ((p1[1]-p2[1])*(p1[1]-p2[1])))

def iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

class Objects:

    def __init__(self, idx, center, dim, frameno):
        self.idx = idx
        self.dim = dim
        self.center = center
        self.frames = []
        self.frames.append(frameno)
        self.isAbandoned = False

    def update_frames(self, frame):
        self.frames.append(frame)
        
    def last_frame(self):
        return self.frames[-1]
        
    def count_consecutive(self):
        return len(self.frames)

    def __str__(self):
        return "id: " + str(self.idx) + " c: " + str(self.center) + " f: " + str(len(self.frames)) + " l: " + str(self.frames[-1])
        
    


def get_object_prediction(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()
    
    # only take the highest confidence ignore rest
    max_confidence = 0
    for i in np.arange(0, detections.shape[2]):
        if detections[0, 0, i, 2]>max_confidence:
            max_confidence = detections[0, 0, i, 2]

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., the probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # skip if not highest confidence
        if confidence < max_confidence:
            continue

        # filter out weak detections by ensuring the 'confidence' is greater than the minimum confidence
        if confidence > min_confidence:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            detected_object = CLASSES[idx]
            flag = False
            if detected_object != "person":
                detected_object = "Suspicious object"
                flag = True
            # display the prediction
            label = '{}: {:.2f}%'.format(detected_object, confidence * 100)
            print(label)

            return flag
    return False
    
frameno = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret==0:
        break
    cv2.imshow('main',frame)
    frame_copy = frame.copy()
    frameno = frameno + 1
    cv2.imwrite("frames/"+str(frameno)+".jpg", frame)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray,(21,21),0)
     
    frame_diff = cv2.absdiff(firstframe, frame)
  
    edged = cv2.Canny(frame_diff,10,200)
    cv2.imshow('CannyEdgeDet',edged)

    kernel2 = np.ones((5,5),np.uint8)
    thresh2 = cv2.morphologyEx(edged,cv2.MORPH_CLOSE, kernel2,iterations=5)
    cv2.imwrite("frames/"+str(frameno)+"_m.jpg", thresh2)
    cv2.imshow('Morph_Close', thresh2)

    (cnts, _) = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    mycnts =[]
    # loop over the contours
    for c in cnts:
        isStored = False
        M = cv2.moments(c)
        if M['m00'] == 0: 
            pass
        else:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            if cv2.contourArea(c) < 250 or cv2.contourArea(c)>20000:
                pass
            else:
                (x, y, w, h) = cv2.boundingRect(c)
                dim = [x, y, x+w, y+h]
                if x <= 3 or dim[2] >= frame.shape[1]-3 or y <= 3 or dim[3] >= frame.shape[0]-3:
                    continue
                mycnts.append(c)
                
                object = Objects(idxGlobal,(cx,cy), dim, frameno)
                # print("Current object: ", object)
                idxGlobal+=1
                # print("List length: ", len(allObjects))
                for o in allObjects:
                    # print("Checking with object: ", o, "\t|\tIOU with above: ", iou(dim, o.dim))
                    if iou(dim, o.dim)>0.4:                     
                        o.update_frames(frameno)
                        object = o
                        isStored = True
                        break
                
                if not isStored:
                    allObjects.append(object)

                if len(object.frames)>100:
                    print(object)
                    crop_img = frame[y-10:y+h+10, x-10:x+w+10]
                    cv2.imshow("Image detection", crop_img)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    cv2.putText(frame,'%s'%('Static object'), (cx,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

                    if get_object_prediction(crop_img):
                        object.isAbandoned = True
                        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        cv2.putText(frame_copy,'%s'%('Abandoned object Detected!'), (cx,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                        print ('Abandoned Detected at: ', cx, cy, 'frame_no: ',frameno)

            
    remove_list = []
   
    print("\n\nAll objects: ", frameno)
    for it in range(len(allObjects)):
        print(allObjects[it])
        if allObjects[it].isAbandoned:
            dim = allObjects[it].dim
            first_not_found = True
            ab_image = cv2.imread("frames/"+str(allObjects[it].frames[0])+"_m.jpg")
            img_idx = 0
            while(first_not_found):
                img_idx += 5
                small_image = ab_image[dim[1]:dim[3], dim[0]:dim[2]]
                cv2.rectangle(ab_image, (dim[0], dim[1]), (dim[2], dim[3]), (0, 0, 255), 3)
                mean = np.mean(small_image)
                ab_image = cv2.imread("frames/"+str(allObjects[it].frames[0]-img_idx)+"_m.jpg")
                if mean<75:
                    first_not_found = False
                    ab_image = cv2.imread("frames/"+str(allObjects[it].frames[0]-img_idx+5)+".jpg")
            boxes, idxs, classIDs, confidences = detector.detect(ab_image)
            cv2.imshow("Abandoned object first frame No. " + str(allObjects[it].frames[0]) +")", ab_image)
            ab2 = ab_image.copy()
            # ensure at least one detection exists
            if len(idxs) > 0:
	            # loop over the indexes we are keeping
	            for i in idxs.flatten():
		            # extract the bounding box coordinates
		            (x, y) = (boxes[i][0], boxes[i][1])
		            (w, h) = (boxes[i][2], boxes[i][3])
		            # draw a bounding box rectangle and label on the image
		            color = (0, 255, 0)
		            cv2.rectangle(ab_image, (x, y), (x + w, y + h), color, 2)
		            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		            cv2.putText(ab_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			            0.5, color, 2)
            # show the output image
            cv2.imshow("YOLO on first frame", ab_image)
            print(boxes)
            
            iou_min = 0          
            culprit = [] 
            for box in boxes:
                iou_current = iou([box[0],box[1],box[0]+box[2],box[1]+box[3]], dim)
                print(iou_current, box)
                if iou_current>iou_min:
                    culprit = box
                    iou_min = iou_current
            cv2.imshow("Owner of abandoned object: ", ab2[culprit[1]:culprit[1]+culprit[3], culprit[0]:culprit[0]+culprit[2]])
            cv2.waitKey(0)
        if frameno-allObjects[it].last_frame()>20:
            remove_list.append(it)
    print("\n")
    it = 0
    for remove in remove_list:
        remove -= it
        it+= 1
        del allObjects[remove]

    cv2.imshow('Static Object Detection',frame)
    cv2.imshow('Abandoned Object Detection',frame_copy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)    
cap.release()
# cv2.destroyAllWindows()

