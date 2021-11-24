import cv2
from cv2 import cv2 # make vscode not complain
from matplotlib import pyplot as plt
import math
import numpy as np
def combine_grad(sx, sy):
    if sx.shape != sy.shape:
        raise ValueError 
    HEIGHT, WIDTH = sx.shape[0], sx.shape[1]
    grad_M = np.zeros([HEIGHT,WIDTH])
    print(grad_M.shape)
    for i in range(HEIGHT):
        for j in range(WIDTH):
            grad_M[i][j] = np.sqrt(np.square(sx[i][j]) + np.square(sy[i][j]) )
    return grad_M

def nonMaxSuppression(M,dx,dy):
    if not (M.shape == dx.shape and M.shape == dy.shape):
        raise ValueError 
    HEIGHT, WIDTH = M.shape[0], M.shape[1]
    EPS = 1e-6
    for i in range(1, HEIGHT - 1):
        for j in range(1, WIDTH - 1):
            gradx = float(dx[i][j])
            grady = float(dy[i][j])
            if np.abs(gradx) < EPS:
                gradx = np.sign(gradx) * EPS
            if np.abs(grady) < EPS:
                grady = np.sign(grady) * EPS
                
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
                    
def docanny(filename):
    SAVEDIR = 'plot/'
    READDIR = 'dataset/'
    img_gray = cv2.imread(READDIR + filename,cv2.IMREAD_GRAYSCALE)
    img_gauss= cv2.GaussianBlur(img_gray,(3,3),0)
    sx = cv2.convertScaleAbs(cv2.Sobel(img_gauss,cv2.CV_16S,1,0))  # x方向梯度
    sy = cv2.convertScaleAbs(cv2.Sobel(img_gauss,cv2.CV_16S,0,1))  # y方向梯度
    M = cv2.convertScaleAbs(combine_grad(sx.astype(int),sy.astype(int)))
    #print(M)
    nonMaxSuppression(M,sx,sy)
    cv2.imshow(filename,M)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    """
    img_grad = calcGradient(img_gray)
    bins = np.array(range(int(np.max(img_grad))))  - 0.01  # 防止值落在边界上
    plt.hist(img_grad.ravel(),bins=bins,density=True,rwidth=1)
    
    plt.title('Gradient Histogtam of ' + filename)
    plt.savefig(SAVEDIR + filename +'_GRAD_Hist.png')
    #plt.show()
    plt.close()
    """
    

if __name__ == '__main__':
    files = [
    '1.jpg',
    '2.jpg',
    '3.jpg'
    ] 
    for file in files:
        docanny(file)  




