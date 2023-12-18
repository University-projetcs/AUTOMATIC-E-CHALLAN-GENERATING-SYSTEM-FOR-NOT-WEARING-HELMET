import cv2
import numpy as np
import os
import imutils
import re
from imutils.video import FPS
from tensorflow.keras.models import load_model
import pytesseract as tess
import glob
 
tess.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['KMP_DUPLICATE_LIB_OK']='true'

net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


model = load_model('helmet-nonhelmet_cnn.h5')
print('model loaded sucessfully!!!')

COLORS = [(0,255,0),(0,0,255)]

plates=[]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


def helmet_or_nohelmet(helmet_roi):
	try:
		helmet_roi = cv2.resize(helmet_roi, (224, 224))
		helmet_roi = np.array(helmet_roi,dtype='float32')
		helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
		helmet_roi = helmet_roi/255.0
		return int(model.predict(helmet_roi)[0][0])
	except:
			pass
files = glob.glob('upload/*')
for file in files:
    k = True
    while k:
        print(file)
        img = cv2.imread(file)
        img = imutils.resize(img,height=500)
        height, width = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 0.00250, (416, 416), (0, 0, 0), True, crop=False)
       
        net.setInput(blob)
        outs = net.forward(output_layers)

        confidences = []
        boxes = []
        classIds = []
        
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.75:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)

                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    classIds.append(class_id)
                    
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.4)
        
        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                color = [int(c) for c in COLORS[classIds[i]]]
                # green --> bike
                # red --> number plate
                if classIds[i]==0: #bike
                    helmet_roi = img[max(0,y):max(0,y)+max(0,h)//4,max(0,x):max(0,x)+max(0,w)]
                else: #number plate
                    x_h = x-75
                    y_h = y-275
                    w_h = w+75
                    h_h = h+75
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                    h_r = img[max(0,(y-330)):max(0,(y-330 + h+100)) , max(0,(x-80)):max(0,(x-80 + w+130))]
                    if y_h>0 and x_h>0:
                        h_r = img[y_h:y_h+h_h , x_h:x_h +w_h]
                        c = helmet_or_nohelmet(h_r)
                        if c == 0:
                            cv2.putText(img,'no-helmet',(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)             
                            # cv2.rectangle(img, (x_h, y_h), (x_h + w_h, y_h + h_h),(255,0,0), 2) #rectangle around face
                            cv2.imshow("Image", img)
                                    
                            k = cv2.resize(img[y:y+h,x:x+w],(300,100))
                            k = cv2.cvtColor(k,cv2.COLOR_BGR2GRAY)
                           
                            k = cv2.GaussianBlur(k,(5,3),0)
                            k = cv2.medianBlur(k,3)
                            
                            edges = cv2.Laplacian(k,cv2.CV_8U , ksize=3)
                            ret,mask = cv2.threshold(edges,100,255,cv2.THRESH_BINARY_INV)

                            cv2.imwrite(f"main.png",k)
                            imag = cv2.imread("main.png")

                            contours, _= cv2.findContours(mask, cv2.RETR_TREE, 1)
                            cnt = contours
                            for i in range(len(contours)):
                                if (cv2.contourArea(cnt[i]) > 1500):
                                    final_image = cv2.drawContours(imag, cnt[i], 0, (0,255,0), 3)
                            cv2.imshow('Marked image', final_image )
                            cv2.waitKey(100)

                            
                            text = tess.image_to_string(final_image)
                            text=re.sub('[^A-Za-z0-9]+','', text)
                            if text not in plates:
                                plates.append(text)
                            k = False
                        
                        else:
                            cv2.putText(img,'helmet',(x,y-50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)                
                            cv2.rectangle(img, (x_h, y_h), (x_h + w_h, y_h + h_h),(255,0,0), 2)
                            cv2.rectangle(img, (x, y), (x + w, y + h), (255,255,0), 1)
                            k = False     
                        k = False 
                    k = False 
                k = False 
 
                        
    print(plates)

    if cv2.waitKey(2) == 27:
        break
cv2.waitKey(100)
cv2.destroyAllWindows