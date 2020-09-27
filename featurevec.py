import cv2
import numpy as np

#images resized to 10*10 for Training
imageNamesTrain = ['Kimg1-resized.png','Kimg2-resized.png','Kimg3-resized.png',
              'Kimg4-resized.png','Kimg5-resized.png','Dimg1-resized.png',
              'Dimg2-resized.png','Dimg3-resized.png','Dimg4-resized.png',
              'Dimg5-resized.png','Mimg1-resized.png','Mimg2-resized.png',
              'Mimg3-resized.png','Mimg4-resized.png','Mimg5-resized.png']

#images resized to 10*10 for testing
imageNamesTest = ['Kimg6-resize.png','Kimg7-resize.png','Dimg6-resize.png',
                  'Dimg7-resize.png','Mimg6-resize.png','Mimg7-resize.png',
                  'Simg1-resize.png','Aimg1-resize.png']



print("/n Feature vectors for Training inputs\n")

for imageName in imageNamesTrain:
    img = cv2.imread(f"images/{imageName}", 0) #get image as a grid of 10*10
    for i in range (10):
        for j in range (10): 
            if img[i][j]==255:                 #make conversion to 0 and 1           
                img[i][j] = 1
            else:
                img[i][j] = 0
    A = np.asarray(img).reshape(-1)           #conacatenate rows
    
    print("[",end="")
    for a in A:
        print(a,end=",")
    print("]")
    print("\n")

print("/n Feature vectors for Testing inputs\n")

for imageName in imageNamesTest:
    img = cv2.imread(f"images/{imageName}", 0) 
    for i in range (10):
        for j in range (10): 
            if img[i][j]==255:
                img[i][j] = 1
            else:
                img[i][j] = 0
    A = np.asarray(img).reshape(-1)
    
    print("[",end="")
    for a in A:
        print(a,end=",")
    print("]")
    print("\n")