from __future__ import print_function
import cv2 as cv
from histograms_and_model import get_hist, get_model
import numpy as np
from sklearn import preprocessing
from joblib import load

def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray)
    
    try:
        x, y, w, h = faces[0]
        draw_rect(x, y, w, w, frame)
        roi_gray=frame_gray[y:y+w,x:x+h]  
        roi_gray=cv.resize(roi_gray,(48,48)) 
        # cv.imshow('Capture - Face detection', frame)    
        return roi_gray, x, y - 10
    except:
        return np.array([None]), 0, 5
        

def draw_rect(x, y, w, h, frame):
    pt1 = (x, y)
    pt2 = (x+w, y+h)
    frame = cv.rectangle(frame, pt1, pt2, (0, 255, 0), thickness=3)
    
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(1)
model = load('model.joblib')

if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    
    face, x, y = detectAndDisplay(frame)
    
    if face.all():
        face = cv.resize(cv.cvtColor(face, cv.COLOR_GRAY2BGR), (48, 48))
    
        X = np.array([get_hist(face)])
        X = preprocessing.normalize(X)
        prediction = model.predict(X)
        text = str(prediction).replace('[\'', '').replace('\']', '')
        cv.putText(frame, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
        cv.imshow('Capture - Face detection', frame)
  
    if cv.waitKey(10) == 27:  # escape closes the window 
        cap.release()
        cv.destroyAllWindows()
        break
