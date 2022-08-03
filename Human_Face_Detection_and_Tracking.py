import cv2
alg = "haarcascade_frontalface_default.txt"          #Algorithm used for face recog
haar_cascade = haar_cascade = cv2.CascadeClassifier(alg)             #Loading the algorithm into cascadee classifier
cam = cv2.VideoCapture(0) #Starting port 1 camera

while True:
    _, img = cam.read() #Get the image
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Convert image to gray
    face = haar_cascade.detectMultiScale(grayImg, 1.3, 4) #Obtain the dimentions of the bounding block
    for (x,y, w, h) in face:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2) #Draw the rectangle
    cv2.imshow("FaceDetection", img)
    key = cv2.waitKey(10)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()

#To access mobile camera using ip webcam
'''
import urlib.request
import cv2
import numpy as np
import imutils

url = 'http://192.168.6.8080/shot.jpg'

while True:
    imgPath = urlib.request.urlopen(url) #Open url
    imNp = np.array(bytearry(imgPath.read()), dtype=np.unit6) #Data is read in form of byres
    img = cv2.imdecode(imgNp, -1) #Taking the data into image
     #Frist three lines to get frame of camera
    img = imutils.resize(img, width = 450) #Resizing
    cv2.imshow("CameraFeed", img) #Give live feed

    if ord('q') == cv2.waitKey(1)
    exit(0)'''