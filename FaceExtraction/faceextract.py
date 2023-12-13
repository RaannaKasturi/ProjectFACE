import cv2
import glob
import os

#initializing model to detect faces
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt_tree.xml')

#error handeling for haarcascade file
if face_cascade.empty():
    print("Error: Unable to load the cascade classifier.")
    exit()

#imagepath
path = "nayan/*.*"
img_number = 1  #Start an iterator for image number.

img_list = glob.glob(path)

#Resize to 128x128
for count, file in enumerate(img_list[0:25000], start=1):
    print(f"Processing image {count} of {len(img_list)}.")
    img= cv2.imread(file, 1)  #now, we can read each file since we have the full path

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    try:
        for (x,y,w,h) in faces:
            roi_color = img[y:y+h, x:x+w] 
        resized = cv2.resize(roi_color, (128,128))
        cv2.imwrite("extracted_faces/"+str(img_number)+".jpg", resized)
        img_number +=1
    except:
        print("No faces detected")
