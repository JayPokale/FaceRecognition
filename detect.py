from imutils import face_utils
import dlib
import cv2
import argparse
from simple_facerec import SimpleFacerec
import math
from math import degrees
from deepface import DeepFace
import numpy as np
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# Initialize face recognition class
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Returns highlighted faces in images
def highlightFace(net, frame, conf_threshold=0.6):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/250)), 8)
    return frameOpencvDnn,faceBoxes


parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

video=cv2.VideoCapture(0)
padding=20

while cv2.waitKey(1)<0:

    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break
    
    resultImg,faceBoxes=highlightFace(faceNet,frame)

    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):min(faceBox[3]+padding,frame.shape[0]-1),
                max(0,faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]
        
        # Face recognition
        name = sfr.detect_known_faces(face)
        if(len(name)): name = name

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

        try:
            # Gender prediction
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]

            # Age prediction
            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]
        except: None

        cv2.putText(resultImg, f'{name}', (faceBox[0], faceBox[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(frame, 0)
    for (i, rect) in enumerate(rects):
        try:
            # Mark facial coordinates
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            points = []
            for (x,y) in shape:
                points.append([x,y])
                cv2.circle(resultImg, (x,y), 2, (0,255,0), -1)
        except: None
        
        def distance(point1, point2):
            return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
        
        # Face shape recognition
        line1 = distance(points[0], points[16])
        line2 = distance(points[2], points[14])
        line3 = distance(points[4], points[12])
        line4 = 2*distance(points[8], points[33])

        similarity = np.std([line1,line2,line3])
        ovalsimilarity = np.std([line2,line4])
        ax,ay = points[3][0],points[3][1]
        bx,by = points[4][0],points[4][1]
        cx,cy = points[5][0],points[5][1]
        dx,dy = points[6][0],points[6][1]

        alpha0 = math.atan2(cy-ay,cx-ax)
        alpha1 = math.atan2(dy-by,dx-bx)
        alpha = alpha1-alpha0
        angle = abs(degrees(alpha))
        angle = 180-angle

        for i in range(1):
            if similarity<10 and angle<160: shape = "Square Shape"
            elif similarity<10 and angle>=160: shape = "Round Shape"
            elif line3>line1 and angle<160: shape = "Triangle Shape"
            elif ovalsimilarity<10: shape = "Diamond Shape"
            elif line4>line2 and angle<160: shape = "Rectangular Shape"
            elif line4>line2 and angle>=160: shape = "Oblong Shape"

        cv2.putText(resultImg, f'{shape}', (points[8][0], points[8][1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)


        face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for x,y,w,h in face:
            img = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 1)
            try:
                # Emotion recognition
                analyze = DeepFace.analyze(frame, actions=['emotion'])
                cv2.putText(resultImg, f'{analyze[0]["dominant_emotion"]}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            except: None

    cv2.imshow("Detecting", resultImg)
