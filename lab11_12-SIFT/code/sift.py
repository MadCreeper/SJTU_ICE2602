import cv2
from cv2 import Sobel, cv2, resize # make vscode not complain
from matplotlib import pyplot as plt
import math
from math import sin, cos, pi
import numpy as np
import copy
import random
PI = 3.1415926

def arctan_2pi(x,y):  # return 0 ~ 2pi arctan of vector (x,y)
    return np.arctan2(y,x) if y > 0 else np.arctan2(y,x) + 2*PI

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

def __sift(filename):
    
    READDIR = 'dataset/'
    img_gray = cv2.imread(READDIR + filename, cv2.IMREAD_GRAYSCALE)
    img_bgr = cv2.imread(READDIR + filename)
    
    pyramid = calc_img_pyramid(img_gray)
    sift_onelayer(pyramid[0])
    
    
    #img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
def sift_onelayer(img_gray):
    #SAVEDIR = 'result/builtin/'
    #maxCorners = 100
    """
   
    """
    H, W = img_gray.shape[0], img_gray.shape[1]
     # Harris 获得角点
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 1, 1)
    corners = [(int(c[0][0]),int(c[0][1])) for c in \
               cv2.goodFeaturesToTrack(image = img_gray, maxCorners = 100, qualityLevel = 0.01, minDistance = 10,  blockSize = 3, useHarrisDetector = True)]
    corner_cnt = len(corners)
    # list of tuples (corner (x,y))
    #print(corners)
    
    #img_marked = mark_corners(img_bgr,corners)
    #img_marked = cv2.drawKeypoints(img_bgr,corners,img_bgr,color=(255,0,255))
    #cv2.imshow("img_marked", img_marked)
    #cv2.waitKey(0)
    #gaussian_pyramid = calc_img_pyramid(img_gray)

       
    #计算梯度
    gx, gy, gm, gtheta = calc_grad(img_gray)
    # 计算主方向矩阵
    main_dir = vote(corners, gm, gtheta, H, W)
    # 计算直方图 4*4*8 （8*45=360）
    
    # 计算描述子
    print(H, W)
    feature = []
    for i in range(corner_cnt):
        #print(H, W)
        val = calc_feature(corners[i], gtheta, main_dir[i], H, W)
        m = sum(k * k for k in val) ** 0.5
        l = [k / m for k in val]
        feature.append(l)
    return feature, corners, corner_cnt
    
def vote(corners, grad, theta, H, W):
    BINSIZE = (H + W) // 80
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

def calc_feature(pos, gradtheta, theta, HEIGHT, WIDTH):
    def _theta(x, y):
        #print(H)#type(x),type(y),type(H),type(W))
        if (x < 0 or x >= HEIGHT) or (y < 0 or y >= WIDTH):
            return 0
        dif = gradtheta[x][y] - theta
        return dif if dif > 0 else dif + 2 * pi

    def _DB_linear(x, y):
        xx, yy = int(x), int(y)
        dy1, dy2 = y-yy, yy+1-y
        dx1, dx2 = x-xx, xx+1-x
        val = _theta(xx,yy)*dx2*dy2 \
                + _theta(xx+1,yy)*dx1*dy2 \
                + _theta(xx,yy+1)*dx2*dy1 \
                + _theta(xx+1,yy+1)*dx1*dy1
        return val

    y0, x0 = pos
    H = np.array([cos(theta), sin(theta)])
    V = np.array([-sin(theta),cos(theta)])

    val = []
    def cnt(x1, x2, y1, y2, xsign, ysign):
        voting = [0 for i in range(9)]
        for x in range(x1, x2):
            for y in range(y1, y2):
                dp = [x * xsign, y * ysign]
                p = H * dp[0] + V * dp[1]
                bin = int((_DB_linear(p[0]+x0, p[1]+y0))//(PI/4) + 1)
                if bin > 8:
                    bin = 8
                voting[bin] += 1
        return voting[1:]

    bins = (HEIGHT + WIDTH) // 160
    for xsign in [-1,1]:
        for ysign in [-1,1]:
            val += cnt(0, bins, 0, bins, xsign, ysign)
            val += cnt(bins, bins*2, 0, bins, xsign, ysign)
            val += cnt(bins, bins*2, bins, bins*2, xsign, ysign)
            val += cnt(0, bins, bins, bins*2, xsign, ysign)
    return val   

def calc_grad(img):
    H, W = img.shape[0], img.shape[1]
    grad_theta = np.zeros((H, W))
    grad_M =  np.zeros((H, W))
    grad_x = cv2.filter2D(img, cv2.CV_16S, np.array([-1,0,1]).reshape(1,3))
    grad_y = cv2.filter2D(img, cv2.CV_16S, np.array([1,0,-1]).reshape(3,1))
    for i in range(H):
        for j in range(W):
            grad_theta[i][j] = arctan_2pi(grad_x[i][j],grad_y[i][j])
            grad_M[i][j] = np.sqrt(np.square(float(grad_x[i][j])) + np.square(float(grad_y[i][j])))
    return grad_x, grad_y, grad_M, grad_theta



def _Merge(img1, img2):
    h1, w1 ,a= np.shape(img1)
    h2, w2 ,a= np.shape(img2)
    if h1 < h2:
        extra = np.array([[[0,0,0] for i in range(w1)] for ii in range(h2-h1)])
        img1 = np.vstack([img1, extra])
    elif h1 > h2:
        extra = np.array([[[0,0,0] for i in range(w2)] for ii in range(h1-h2)])
        img2 = np.vstack([img2, extra])
    return np.hstack([img1,img2])


def _Match(threshold):
    for id in range(len(imgset)):
        x = []
        cnt = 0
        for i in range(lt):
            tmp = []
            for j in range(ll[id]):
                sc= np.inner(np.array(ft[i]), np.array(ff[id][j]))
                tmp.append(sc)
            x.append([tmp.index(max(tmp)), max(tmp)])
        for a in range(len(x)):
            b, s = x[a]
            if s < threshold:
                continue
            cnt += 1
            color = ((random.randint(0, 255)),
                     (random.randint(0, 255)),
                     (random.randint(0, 255)))
            cv2.line(mgimgs[id], tuple(ct[a]),
                     tuple([cc[id][b][0] + w,
                            cc[id][b][1]]), color, 1)
        if cnt > 6:
            cv2.imwrite("match%d.jpg" % id, mgimgs[id])
            print("MATCHED %d" % id)
            img = np.array(mgimgs[id], dtype="uint8")
            cv2.namedWindow("MATCH_RESULT")
            cv2.imshow("MATCH_RESULT", img)
            cv2.waitKey(0)
            cv2.destroyWindow("MATCH_RESULT")

        else:
            print("NOT %d" % id)

    return

def _Gray(img):
    x, y, z = np.shape(img)
    gray = np.zeros([x,y], "uint8")# 2 unsigned int 8
    for i in range(x):
        for j in range(y):
            gray[i][j] = np.dot(np.array(img[i][j],
                dtype="float"), [.114, .587, .299])
    return gray

if __name__ == "__main__":

    ### SIFT ###
    tgt0 = cv2.imread(r"target.jpg", 1)
    imgset0 = [cv2.imread("%d.jpg" % i, 1) for i in range(3,4)]
    r0,c0,a0=np.shape(tgt0)
    times=1.0
    resized_tgt0=cv2.resize(tgt0,(int(r0*times),int(c0*times)))
    # 灰度化
    tgt = _Gray(resized_tgt0)
    imgset = [_Gray(imgset0[i]) for i in range(len(imgset0))]

    ff = []
    cc = []
    ll = []
    ft, ct, lt = sift_onelayer(tgt)
    for i in range(len(imgset)):
        f, c, l = sift_onelayer(imgset[i])
        ff.append(f)
        cc.append(c)
        ll.append(l)

    w = np.shape(tgt)[1]
    mgimgs = [_Merge(tgt0, imgset0[i]) for i in range(len(imgset0))]
    print("All Original Pics Processed!")

    _Match(0.8)
