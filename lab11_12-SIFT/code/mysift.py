import cv2
from cv2 import Sobel, cv2, resize # make vscode not complain
from matplotlib import pyplot as plt
import math
import numpy as np
import copy
import random

from numpy.matrixlib import mat
PI = 3.1415926

def __atan(x, y):  # return 0 ~ 2PI arctan of vector (x,y)
    return np.arctan2(y, x) if y > 0 else np.arctan2(y,x) + 2*PI

def calc_img_pyramid(img, layers=5): # 计算高斯金字塔
    pyramid = []
    height, width = img.shape[0], img.shape[1]
    multipliers = [int(2 ** i) for i in range(layers)] # 0(original), /2, /4 , etc.
    pyramid.append(img)
    for i in range(1, layers):
        resized_img = cv2.resize(img,(width // 2**i, height // 2**i))
        pyramid.append(resized_img)

    return pyramid, multipliers

    
def sift_singlelayer(img_gray): # 单层图片的sift

    H, W = img_gray.shape[0], img_gray.shape[1]
     # Harris 获得角点
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 1, 1)
    corners = [[int(c[0][0]),int(c[0][1])] for c in \
               cv2.goodFeaturesToTrack(image = img_gray, maxCorners = 50, qualityLevel = 0.01, minDistance = 10,  blockSize = 3, useHarrisDetector = True)]
    corners = np.array(corners)
    corner_cnt = len(corners)
    # list of tuples (corner (x,y))
    #计算梯度
    gx, gy, gm, gtheta = calc_grad(img_gray)
    # 计算主方向矩阵
    main_dir = vote(corners, gm, gtheta, H, W)
    
    # 计算描述子
    #print(H, W)
    descriptors = []
    for i in range(corner_cnt):
        #print(H, W)
        desc = calc_descriptor(corners[i],  gtheta, main_dir[i], H, W)
        descriptors.append(desc / np.linalg.norm(desc)) # 归一化
        
    return descriptors, corners
    
def vote(corners, grad, theta, H, W):  # 主方向 vote直方图
    BINSIZE = (H + W) // 80   # 经测试比固定BINSIZE效果好
    
    main_dir = []
    for corner in corners:
        _hist = np.zeros(36)
        y, x = corner
        for i in range( max(0, x - BINSIZE), min (H, x + BINSIZE + 1)):
            for j in range( max(0, y - BINSIZE), min(W, y + BINSIZE)):
                deg10 = min( round(theta[i][j] * 18 / PI), 35 )
                _hist[deg10] += grad[i][j]
        most_deg10 = np.argmax(_hist)
        main_dir.append(( most_deg10 + 0.5 ) / 18 * PI)
    
    return main_dir


def calc_descriptor(pos, gradtheta, theta, HEIGHT, WIDTH): # 计算一个特征点的描述子
    
    def dbl_linear(x, y): # 双线性插值
        def dtheta(x, y): # delta-theta of vector (x,y) and theta
            if (x < 0 or x >= HEIGHT) or (y < 0 or y >= WIDTH):
                return 0
            diff = gradtheta[x][y] - theta
            return diff if diff > 0 else diff + 2 * PI
        
        xx, yy = int(x), int(y)
        dy1, dy2 = y-yy, yy+1-y
        dx1, dx2 = x-xx, xx+1-x
        interpol = dtheta(xx, yy)*dx2*dy2 \
                + dtheta(xx+1,yy)*dx1*dy2 \
                + dtheta(xx, yy+1)*dx2*dy1 \
                + dtheta(xx+1, yy+1)*dx1*dy1
        return interpol

    y0, x0 = pos
    # 坐标旋转矩阵
    rotation = np.array([[math.cos(theta), - math.sin(theta)],
                        [math.sin(theta), math.cos(theta)]])

    
    def _vote(x1, x2, y1, y2, xsign, ysign):
        hist = np.zeros(8)
        for x in range(x1, x2):
            for y in range(y1, y2):
                v = np.array([x * xsign, y * ysign]).T
                _v = rotation @ v  # 旋转以后的坐标
                deg45 = int( (dbl_linear(_v.T[0] + x0, _v.T[1] + y0)) // (PI/4)) # 分成8份 8*45=360
                hist[min(deg45, 7)] += 1
        return list(hist)
 
    BINSIZE = (HEIGHT + WIDTH) // 128  # 经测试比固定BINSIZE效果好
   
    descriptor = []
    for xsign in [-1,1]:  # 四个象限统计，每个象限细分为4块
        for ysign in [-1,1]:
            descriptor += _vote(0, BINSIZE, 0, BINSIZE, xsign, ysign)
            descriptor += _vote(BINSIZE, BINSIZE * 2, 0, BINSIZE, xsign, ysign)
            descriptor += _vote(BINSIZE, BINSIZE * 2, BINSIZE, BINSIZE * 2, xsign, ysign)
            descriptor += _vote(0, BINSIZE, BINSIZE, BINSIZE * 2, xsign, ysign)
    return np.array(descriptor)


def calc_grad(img): # 计算梯度和梯度方向角度矩阵
    H, W = img.shape[0], img.shape[1]
    grad_theta = np.zeros((H, W))
    grad_M =  np.zeros((H, W))
    grad_x = cv2.filter2D(img, cv2.CV_16S, np.array([-1,0,1]).reshape(1,3))
    grad_y = cv2.filter2D(img, cv2.CV_16S, np.array([1,0,-1]).reshape(3,1))
    for i in range(H):
        for j in range(W):
            grad_theta[i][j] = __atan(grad_x[i][j],grad_y[i][j])
            grad_M[i][j] = np.sqrt(np.square(float(grad_x[i][j])) + np.square(float(grad_y[i][j])))
    return grad_x, grad_y, grad_M, grad_theta


def compare(img, merged, target_desc, target_corners, H, W, filename):
    img_desc, img_corners = sift_singlelayer(img)
    match_2imgs(merged, img_desc, target_desc, img_corners, target_corners, H, W, filename)
    
def sift_multilayer(img): #在图像金字塔上进行sift操作
    pyramid, multipliers = calc_img_pyramid(img)
    i = 0
    all_desc = []
    all_corners = []
    for layer in pyramid:
        desc, corners = sift_singlelayer(layer)
        corners *= multipliers[i] # 把角点坐标变回原来的尺度
        i += 1
        all_desc += desc
        all_corners += list(corners)
    
    return all_desc, all_corners
    
def concatenate_2imgs(img1, img2): # 把两张图横向拼接
    H1, W1 = img1.shape[0:2]
    H2, W2 = img2.shape[0:2]
    
    if H1 == H2:
        return np.hstack([img1, img2]).astype(np.uint8)
    if H1 < H2: # 两张图不一样高的情况
        filler = np.zeros((H2-H1, W1, 3), dtype=int)
        return np.hstack([np.vstack([img1, filler]), img2]).astype(np.uint8)
    else:
        filler = np.zeros((H1-H2, W2, 3), dtype=int)
        return np.hstack([img1, np.vstack([img2, filler])]).astype(np.uint8)


def match_2imgs(merged, desc1, desc2, corners1, corners2, H, W, filename, thresh=0.8, matched_thresh=0.1):
    len1, len2 = len(corners1), len(corners2)
    matched = 0
    for i in range(len1):
        for j in range(len2):
            if np.inner(desc1[i], desc2[j]) > thresh:
                matched += 1
                 # 匹配点画圆圈和线
                color = color = ((random.randint(0, 255)), (random.randint(0, 255)), (random.randint(0, 255))) # random color
                pos1 = tuple([int(corners2[j][0]), int(corners2[j][1])])
                pos2 = tuple([int(corners1[i][0] + W), int(corners1[i][1])])
                cv2.line(merged, pos1, pos2, color=color, thickness=1)
                cv2.circle(merged, pos1, radius=5, color=color, thickness=2)
                cv2.circle(merged, pos2, radius=5, color=color, thickness=2)
    
    if matched > matched_thresh * min(len1, len2): # 匹配点数量大于10%，认为匹配到了
        cv2.imwrite(f"result_{filename}_thresh{thresh}.png", merged)
        print(f"{filename} Match! Showing result: {matched} corners matched")
        cv2.imshow(f"{filename}", merged)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print(f"{filename} NO Match!")


if __name__ == "__main__":

    target = cv2.imread(r"target.jpg")
    imgs = [cv2.imread(f"./dataset/{i}.jpg") for i in range(1,8)]
    
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]  # 转化为灰度图

 
    target_desc, target_corners = sift_multilayer(target_gray)
    H, W = target.shape[0:2]
    for i, img_gray in enumerate(imgs_gray):
        merged = concatenate_2imgs(target, imgs[i])
      
        compare(img_gray, merged, target_desc, target_corners, H, W, str(i+1)+'.jpg')

