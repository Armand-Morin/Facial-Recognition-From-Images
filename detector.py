import cv2
import numpy as np
import os

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
Name = 'unknown'
id = 0
font = cv2.FONT_HERSHEY_SIMPLEX


def detection(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.15, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        id, conf = recognizer.predict(gray[y:y + h, x:x + w])
        if id == 1:
            Name = 'Cardi B'
        if id == 2:
            Name = 'Chirac'
        if id == 3:
            Name = 'Giroud'
        if id == 4:
            Name = 'Hollande'
        if id == 5:
            Name = 'Macron'
        if id == 6:
            Name = 'Ninho'
        if id == 7:
            Name = 'Sarkozy'
        if id == 8:
            Name = 'Swift'
        if id == 9:
            Name = 'Trump'
        cv2.putText(img, Name, (x, y + h), font, 3, (255, 0, 0), 3)
    cv2.imshow('frame', img)
    cv2.waitKey(1000)

path='test_images'
imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
for imagePath in imagePaths:
    detection(imagePath)
cv2.destroyAllWindows()
