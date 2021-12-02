import cv2
from cv2 import Sobel, cv2, resize # make vscode not complain
from matplotlib import pyplot as plt
import math
import numpy as np
import copy
def calc_img_pyramid(img, k=5):
    pyramid = []
    height, width = img.shape[0], img.shape[1]
    print(height, width)
    for i in range(k):
        resized_img = cv2.resize(img,( int(width / 2**i),int(height / 2**i) ) )
        pyramid.append(resized_img)
        cv2.imshow(str(i),resized_img)
        cv2.waitKey(0)
    return pyramid    

def mark_corners(img,corners):
    marked = copy.deepcopy(img)
    for corner in corners:
        x, y = int(corner[0][0]), int(corner[0][1])
        #print(x,y)
        cv2.circle(marked, (x,y),radius=5,color= [0,0,255])
    return marked

def corner_detection(filename):
    #SAVEDIR = 'result/builtin/'
    #maxCorners = 100
    READDIR = 'dataset/'
    img_gray = cv2.imread(READDIR + filename, cv2.IMREAD_GRAYSCALE)
    img_bgr = cv2.imread(READDIR + filename)
    #img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  
    corners = cv2.goodFeaturesToTrack(image = img_gray, maxCorners = 100, qualityLevel = 0.01, minDistance = 10,  blockSize = 3, useHarrisDetector = True)
    img_marked = mark_corners(img_bgr,corners)
    #img_marked = cv2.drawKeypoints(img_bgr,corners,img_bgr,color=(255,0,255))
    cv2.imshow("img_marked", img_marked)
    cv2.waitKey(0)
    calc_img_pyramid(img_gray)
    
if __name__ == '__main__':
    files = [
    #'1.jpg'
    #'2.jpg',
    #'3.jpg'
    #'4.jpg',
    #'5.jpg'
    "test.png"
    ] 
    for file in files:
       corner_detection(file)

