import cv2
from cv2 import cv2 # make vscode not complain
from matplotlib import pyplot as plt
import math
import numpy as np
import copy
def combine_grad(sx, sy):
    if sx.shape != sy.shape:
        raise ValueError 
    HEIGHT, WIDTH = sx.shape[0], sx.shape[1]
    grad_M = np.zeros([HEIGHT,WIDTH])
    print(grad_M.shape)
    for i in range(HEIGHT):
        for j in range(WIDTH):
            grad_M[i][j] = np.sqrt( sx[i][j] *  sx[i][j] + sy[i][j] * sy[i][j] )
    
    grad_M = grad_M * 255 / np.max(grad_M)
    return grad_M

def nonMaxSuppression(img,dx,dy):
    M = copy.deepcopy(img)
    if not (M.shape == dx.shape and M.shape == dy.shape):
        raise ValueError 
    HEIGHT, WIDTH = M.shape[0], M.shape[1]
    EPS = 1e-6
    for i in range(1, HEIGHT - 1):
        for j in range(1, WIDTH - 1):
            gradx = float(dx[i][j])
            grady = float(dy[i][j])
            if np.abs(gradx) < EPS:
                gradx = EPS
            if np.abs(grady) < EPS:
                grady = EPS
                
            #gradm = M[i][j]
            if np.abs(grady) > np.abs(gradx):  # 梯度绝对值y方向大于x方向
                w = np.abs(gradx) / np.abs(grady)
                g2 = M[i-1][j]
                g4 = M[i+1][j]
                
                if gradx * grady > 0: # x,y梯度同号
                    g1 = M[i-1][j-1]
                    g3 = M[i+1][j+1]
                else: # x,y梯度反号
                    g1 = M[i-1][j+1]
                    g3 = M[i+1][j-1]
            else:
                w =  np.abs(grady) / np.abs(gradx)
                g2 = M[i][j-1]
                g4 = M[i][j+1]
                
                if gradx * grady > 0: # x,y梯度同号
                    g1 = M[i-1][j-1]
                    g3 = M[i+1][j+1]
                else: # x,y梯度反向
                    g1 = M[i-1][j+1]
                    g3 = M[i+1][j-1]
            
            gtmp1 = w * g1 + (1 - w) * g2
            gtmp2 = w * g3 + (1 - w) * g4
            if M[i][j] < gtmp1 or M[i][j] < gtmp2:
                M[i][j] = 0    
    return M

def doubleThresh(M,low_th,hi_th):
    img = copy.deepcopy(M)
    HEIGHT, WIDTH = img.shape[0], img.shape[1]
    for i in range(1, HEIGHT - 1):
        for j in range(1, WIDTH - 1):
            if img[i][j] < low_th:
                img[i][j] = 0
            elif img[i][j] > hi_th:
                img[i][j] = 255
            elif (img[i-1][j-1:j+1] < hi_th).any() or (img[i+1][j-1:j+1] < hi_th).any() or (img[i][j-1] < hi_th) or (img[i][j+1] < hi_th): # 8邻域中有小于hi_th的点
                img[i][j] = 255
    return img

def docanny(filename):
    SAVEDIR = 'result/'
    READDIR = 'dataset/'
    lo_th, hi_th = 60, 150

    img_gray = cv2.imread(READDIR + filename, cv2.IMREAD_GRAYSCALE)
    img_gauss= cv2.GaussianBlur(img_gray,(3,3),0)
    sx = cv2.convertScaleAbs(cv2.Sobel(img_gauss, cv2.CV_16S, 1, 0))  # x方向梯度
    sy = cv2.convertScaleAbs(cv2.Sobel(img_gauss, cv2.CV_16S, 0, 1))  # y方向梯度
    M = cv2.convertScaleAbs(combine_grad(sx.astype(int), sy.astype(int)))
    #print(M)
    img_sup = nonMaxSuppression(M, sx, sy)
    
    cv2.imshow(filename, M)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    img_fin = doubleThresh(img_sup, lo_th, hi_th)

    cv2.imwrite(f"{SAVEDIR}/lowTH{lo_th}_hiTH{hi_th}_{filename}", img_fin)
    """
    cv2.imshow(filename, img_fin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

def docanny_builtin(filename):
    SAVEDIR = 'result/'
    READDIR = 'dataset/'
    lo_th, hi_th = 60, 150
    img_gray = cv2.imread(READDIR + filename, cv2.IMREAD_GRAYSCALE)
    img_fin = cv2.Canny(img_gray, lo_th, hi_th)
    cv2.imwrite(f"{SAVEDIR}/lowTH{lo_th}_hiTH{hi_th}_builtin_{filename}", img_fin)

if __name__ == '__main__':
    files = [
    '1.jpg',
    '2.jpg',
    '3.jpg'
    ] 
    for file in files:
        docanny(file)  
        docanny_builtin(file)




