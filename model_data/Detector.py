from sre_constants import SUCCESS
import cv2
import numpy as nmy
import time

nmy.random.seed(20)
class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(350,350)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean([127.5,127.5,127.5])
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0, '__Background__')

        self.colorList = nmy.random.uniform(low=0,high=255,size=(len(self.classesList),3))
        #print(self.classesList)

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if (cap.isOpened()==False):
            print("Error opening a file")
            return

        (success, image) = cap.read()

        startTime = 0

        while success:
            currentTime = time.time()
            fps = 1/(currentTime - startTime)
            startTime = currentTime
            classlabelIds, confidences, bboxs = self.net.detect(image, confThreshold = 0.5)

            bbosx = list(bboxs)
            confidences = list (nmy.array(confidences).reshape(1,-1)[0])
            confidences =list(map(float,confidences))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_Threshold = 0.5, nms_threshold = 0.2)

            if len(bboxIdx) != 0:
                for i in range(0,len(bboxIdx)):
                    bbox= bboxs [nmy.squeeze(bboxIdx[i])]
                    classConfidence = confidences [nmy.squeeze(bboxIdx[i])]
                    classlabelIds = nmy.squeeze(classlabelIds[nmy.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classlabelIds]
                    classColor = [int(c) for c in self.colorList[classlabelIds]]
                    
                    displayText = "{}:{:.2f}".format(classLabel, classConfidence)
                    x,y,w,h = bbox

                    cv2.rectangle (image, (x , y), (x + w, y + h), color = classColor , thickness = 1)
                    cv2.putText (image, displayText, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

                    ## for bounding box
                    lineWidth = min(int(w * 0.3), int(h * 0.3))
                    ## top corners 
                    cv2.line(image, (x , y), (x + lineWidth, y), classColor, thickness = 5)
                    cv2.line(image, (x , y), (x, y + lineWidth), classColor, thickness = 5)

                    cv2.line(image, (x + w, y), (x + w - lineWidth, y), classColor, thickness = 5)
                    cv2.line(image, (x + w,y), (x, y + lineWidth), classColor, thickness = 5)

                    ## bottom corners
                    cv2.line(image, (x , y + h), (x + lineWidth, y + h), classColor, thickness = 5)
                    cv2.line(image, (x , y + h), (x, y + h - lineWidth), classColor, thickness = 5)

                    cv2.line(image, (x + w,y + h), (x + w - lineWidth, y + h), classColor, thickness = 5)
                    cv2.line(image, (x + w,y + h), (x + w, y + h - lineWidth), classColor, thickness = 5)

            cv2.putText(image, "FPS: " + str(int(fps)), (20 , 70), cv2.FONT_HERSHEY_PLAIN, 2, (0 , 255 , 0) , 2)
            cv2.imshow("Result" , image) 

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            (success, image) = cap.read()

        cv2.destroyAllWindows() 



