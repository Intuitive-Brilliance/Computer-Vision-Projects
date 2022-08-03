#Importing libraries
import cv2
import time
import imutils

cam = cv2.VideoCapture(0) #Deploy camera at port 1
time.sleep(1) #Provide time delay for 1 second

first_frame = None # To compare the changes in the frame
area = 500

while True: #Infinite loop
    _,img = cam.read() #Take the image from cam and store in img
    text = 'Normal'
    img = imutils.resize(img, width=500)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussianImg = cv2.GaussianBlur(grayImg, (21,21), 0)

    if first_frame is None:
        first_frame = gaussianImg
        continue

    imgDiff = cv2.absdiff(first_frame, gaussianImg)              #difference between the 2 images
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1] #Sets the contrast of the blacka nd white image
    threshImg = cv2.dilate(threshImg, None, iterations= 2) #Dilates thins the image to remove holes
    cnts= cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#Finds the neighbourhood pixels to check the idffernece
    cnts = imutils.grab_contours(cnts) #Creates the boundary by connecting the dots

    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c) #Gets the dimensions for the rectangle around the object
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255, 0), 2)
        text = "Moving Object Detected"
        print(text)
    #If there isnt any change this text at 10,20 location of hershy font 0.5 size amd some color and 2 thicknessis given
    cv2.putText(img, text, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255),2)

    cv2.imshow("cameraFeed", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release() #Turn off camera
cv2.destroyAllWindows() #Close all windows



