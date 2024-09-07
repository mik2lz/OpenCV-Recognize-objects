import cv2
img = cv2.imread('images/elephant.jpg')

classnames = []
classfile = 'files/thing.names'                         #Folder Name

with open(classfile , 'rt') as f :                       # Read The rt File
    classnames = f.read().rstrip('\n').split('\n')       
p = 'files/frozen_inference_graph.pb'                          # Algorithms
v = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'


net = cv2.dnn_DetectionModel(p,v)      # This code opens the file p, v and works on them. It compares the image and opens a rectangle.

net.setInputSize(320,230)              # width and height

net.setInputScale(1.0/127.5)                      # Scale

net.setInputMean((127.5, 127.5, 127.5))           # Constant variables

net.setInputSwapRB(True)                          # Colors



classids, confs, bbox = net.detect(img, confThreshold = 0.1) 
#print(classids, bbox)


for classid , confidence , box in zip(classids.flatten(), confs.flatten(), bbox) :
    cv2.rectangle (img, box, color = (0 , 255 , 0) , thickness = 3)
    cv2.putText(img,classnames[classid-1],
                (box[0] + 10, box[1] + 20) , 
                cv2.FONT_HERSHEY_COMPLEX_SMALL ,1, (0 , 0 , 255), thickness = 2)




cv2.imshow('mikey program', img)
cv2.waitKey(0)