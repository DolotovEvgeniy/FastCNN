import cv2
import numpy as np
import sys
import random 

def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

def intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0:
        return (0, 0, 0, 0) # or (0,0,0,0) ?
    return (x, y, w, h)

filePath = sys.argv[1] 
prefix = sys.argv[2]
dataFile = open(filePath, 'r')

trainList = []
testList = []
testPositiveIndex = 1
testNegativeIndex = 1
trainPositiveIndex = 1
trainNegativeIndex = 1


trainPositiveFile = open("train_positive_fastcnn.txt", 'w')
testPositiveFile = open("test_positive_fastcnn.txt", 'w')

trainNegativeFile = open("train_negative_fastcnn.txt", 'w')
testNegativeFile = open("test_negative_fastcnn.txt", 'w')

home = "/home/dolotov_e/lib/caffe/build/tools/"
while True:
    imageName = dataFile.readline().strip('\n')
    if imageName == "":
        break

    imagePath = prefix + imageName + ".jpg"
    print imagePath

    objectCount = int(dataFile.readline())
    #print objectCount
    
    image = cv2.imread(imagePath)
    rectList = []
    sampleCount = 0
    for i in range(0, objectCount):
        line = dataFile.readline()
        p = line.split(" ")
        x = int(p[0])
        y = int(p[1])
        w = int(p[2])
        h = int(p[3])
        rectList.append((x, y, w, h))
        if 0 < x < image.shape[1]-w and 0 < y< image.shape[0]-h and w > 10 and h > 10: 
            sampleCount += 1
            if random.random() > 0.2 :
                trainPositiveFile.write(home+"train_fastcnn/positive/"+str(trainPositiveIndex)+".jpg"+" 1\n")
                face = image[y:y+h, x:x+w]
                face = cv2.resize(face, (32,32))
                cv2.imwrite("train_fastcnn/positive/"+str(trainPositiveIndex)+".jpg", face)
                trainPositiveIndex += 1
            else : 
                testPositiveFile.write(home+"test_fastcnn/positive/"+str(testPositiveIndex)+".jpg"+" 1\n")
                face = image[y:y+h, x:x+w]
                face = cv2.resize(face, (32,32))
                cv2.imwrite("test_fastcnn/positive/"+str(testPositiveIndex)+".jpg", face)
                testPositiveIndex += 1
    for i in range(0, 3*sampleCount):
        sampleX = random.randint(0, image.shape[1]-33)
        sampleY = random.randint(0, image.shape[0]-33)
        rect = (sampleX, sampleY, 32, 32)
        intersect = False
        for j in range(0, len(rectList)) :
            if intersection(rect, rectList[j]) != (0,0,0,0) :
                intersect = True
        if intersect == False :
            if random.random() > 0.2 :
                trainNegativeFile.write(home+"train_fastcnn/negative/"+str(trainNegativeIndex)+".jpg"+" 0\n")
                face = image[sampleY:sampleY+32, sampleX:sampleX+32] 
                cv2.imwrite("train_fastcnn/negative/"+str(trainNegativeIndex)+".jpg", face)
                trainNegativeIndex += 1
            else :
                testNegativeFile.write(home+"test_fastcnn/negative/"+str(testNegativeIndex)+".jpg"+" 0\n")
                face = image[sampleY:sampleY+32, sampleX:sampleX+32] 
                cv2.imwrite("test_fastcnn/negative/"+str(testNegativeIndex)+".jpg", face)
                testNegativeIndex += 1
