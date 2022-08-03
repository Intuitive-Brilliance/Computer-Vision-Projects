import cv2, numpy, os #cv2 image processing, numpy for array, os for directory operations
haar_file = 'haarcascade_frontalface_default.xml' #Face detection algorithm
face_cascade = cv2.CascadeClassifier(haar_file) #Loading the algorithm
datasets = 'datasets' #Folder
print('Training...')
(images, labels, names, id) = ([], [], {}, 0) #


for (subdirs, dirs, files) in os.walk(datasets): #Acessing in the dataset folder, subfolders, files
    for subdir in dirs: #The subdirectories in main dataset folder
        names[id] = subdir #Assigning ids to the subdirectories
        subjectpath = os.path.join(datasets, subdir) #Setting path of subdirects
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename #Setting up path of the files in subdirects
            label = id #Assigning labels to the files of the same folder
            images.append(cv2.imread(path, 0)) #Data of images is stored in images
            labels.append(int(label)) #Labels are added to the list
        id +=1


(images, labels) = [numpy.array(lis) for lis in [images, labels]] #Converting the lists into array
print(images, labels)
(width, height) = (130, 100)
model = cv2.face.LBPHFaceRecognizer_create() # Face recognition algorithm #1
#model =  cv2.face.FisherFaceRecognizer_create() #Face recognition algorithm #2

model.train(images, labels) #Training the model using the images and labels

webcam = cv2.VideoCapture(2) #Start the camera
cnt=0

while True:
    (_, im) = webcam.read() #Take the image
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #Convert to black and white ie. the gray scale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #Face detection and coordinate acquisition
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,0),2) #Draw rectangle around the face
        face = gray[y:y + h, x:x + w] #Considering only the face in grayscale image
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize) #Passing the resized image and predicting the given image
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        if prediction[1]<800: #Only if the prediction value, ie accuracy and precision is greater than a certain value do we consider it
            cv2.putText(im,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,2,(0, 0, 255))
            print (names[prediction[0]])
            cnt=0 #If an already trained image is detected
        else:
            cnt+=1 #If something else is detected
            #Displaying text above the rectangle
            cv2.putText(im,'Unknown',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            if(cnt>100): #Only when it is detected more than 100 frames
                print("Unknown Person")
                cv2.imwrite("unKnown.jpg",im)
                cnt=0
    cv2.imshow('FaceRecognition', im)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()



