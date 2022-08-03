import imutils
import cv2
#Values for red color
redLower = (87, 136, 145)
redUpper = (177, 255, 255)

camera = cv2.VideoCapture(0)#Sart camera at port 0

while True:
    (grabbed, frame) = camera.read() #Get the feed
    frame = imutils.resize(frame, width = 600) #Resize the image to fit the frame
    blurred = cv2.GaussianBlur(frame, (11,11), 0) #Smoothening the image using gaussuan bkr
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) #Converting BGR image into HSV image, Hue Saturation value

    mask = cv2.inRange(hsv, redLower, redUpper) #Mask or isolate the given color using the values specified
    mask = cv2.erode(mask, None, iterations = 2) #Remove noise
    mask = cv2.dilate(mask, None, iterations = 2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]#Find the contour if the object with the given color is available
    #If cnt count value is higher, meaning there is presence of red color and vice versa
    centre = None
    if len(cnts)>0: #If contour is present
        c = max(cnts, key=cv2.contourArea) #Maximum contour area is found
        #Drawing the minimum enclosure circle anywhere the color is present
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        #To draw a circle around the entire object, centre needs to be found out
        M = cv2.moments(c) #To find centre moments about the area needs to be found
        centre = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
        if radius >10: #Drawing circle only if the object is of specified size
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255),2) #Drawing the circle
            cv2.circle(frame, centre, 5, (0, 0, 255),-1) #Drawing the centre point
            print(centre,radius)
            if radius>250: #The object is far away
                print("stop")
            else:
                if(centre[0]<150):#Only x coordinate is considered. If the centre moves by certain distane indicate
                    print("Left")
                elif(centre[0]>450):
                    print("Right")
                elif(radius<250):
                    print("Front")
                else:
                    print("Stop")

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) &0xFF
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()