from __future__ import print_function
from os import getcwd, listdir, chdir
import cv2 as cv
import numpy as np

def detect(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    try:
        x, y, w, h = faces[0]
        #draw_rect(x, y, w, w, frame)
        roi_gray=frame_gray[y:y+w,x:x+h]  
        roi_gray=cv.resize(roi_gray,(48,48))     
        return roi_gray
    except:
        return np.array([None])

def draw_rect(x, y, w, h, frame):
    pt1 = (x, y)
    pt2 = (x+w, y+h)
    frame = cv.rectangle(frame, pt1, pt2, (0, 255, 0), thickness=3)
    
def get_filenames(emotion):
    path = '%s\\Nasza baza\\%s' % (getcwd(), emotion)
    filenames = [f for f in listdir(path) if f.endswith('.jpg')]
    return filenames
    
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

xyz = getcwd()
path = '%s\\Nasza baza' % (getcwd())
newPath = '%s\\Nasza baza2' % (getcwd())
emotions = [directories for directories in listdir(path)]
for emotion in emotions:
    chdir(xyz)
    filenames = get_filenames(emotion)
    directory = newPath + '\\' + emotion
    for filename in filenames:
        full_path = '%s\\%s\\%s' % (path, emotion, filename)
        print(full_path)
        image = cv.imread(full_path)
        img = detect(image) 
        chdir(directory)
        cv.imwrite(filename, img)
        