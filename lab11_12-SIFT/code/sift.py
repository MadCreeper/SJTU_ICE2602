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

def sift(filename):
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
    gaussian_pyramid = calc_img_pyramid(img_gray)

    for layer in gaussian_pyramid:
        pass

def calc_grad(img):
    H, W = img.shape[0], img.shape[1]
    img_theta = np.zeros((H, W))
    grad_x = cv2.filter2D(img, cv2.CV_16S, np.array([-1,0,1]).reshape(1,3))
    grad_y = cv2.filter2D(img, cv2.CV_16S, np.array([1,0,-1]).reshape(3,1))
    for i in range(H):
        for j in range(W):
            img_theta = theta_grad(grad_x[i][j],grad_y[i][j])
def calc_point_Descriptor():
    pass

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
       sift(file)

