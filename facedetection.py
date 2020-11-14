import cv2
def face_detection(image):
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img=cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.1,4)
    for(x,y,w,h)in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('img',img)
    cv2.waitKey(1000)
cv2.destroyAllWindows()
face_detection('/Users/armand_morin/PycharmProjects/CW/reconnaissance-visuelle-gpe-15-readmeArmand/facedetection_from_images/dataSet/3.5.jpg')
