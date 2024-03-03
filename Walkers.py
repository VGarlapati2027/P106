import cv2

body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')
cap = cv2.VideoCapture('walking.avi')


grey = cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY)
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
bodies = body_classifier.detectMultiScale(grey)
for x,y,w,h in bodies:
    cv2.rectangle(cap,(x,y),(x+w,y+h),(255,0,0),2)