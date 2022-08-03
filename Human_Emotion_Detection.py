from facial_emotion_recognition import EmotionRecognition #Importing emotionrecog method from the library

import cv2 as cv

er = EmotionRecognition(device='cpu') #Deploying emotionrecog algorithm
cam = cv.VideoCapture(0) #Turn on video

while True:
    success, frame = cam.read() #Take image
    frame = er.recognise_emotion(frame, return_type='BGR') #Passing the captured image and running it through the algorithm
    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key == 27:
        break

cam.release()
cv.destroyAllWindows()