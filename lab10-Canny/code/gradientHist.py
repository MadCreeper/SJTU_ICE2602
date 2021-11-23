import cv2
from cv2 import cv2 # make vscode not complain
from matplotlib import pyplot as plt
import numpy as np
def calcGradient(img):
    HEIGHT, WIDTH = img.shape[0], img.shape[1]
    img_grad = np.zeros((HEIGHT - 2,WIDTH - 2))
    for x in range(1,HEIGHT - 1):
        for y in range(1,WIDTH - 1):
            grad_x = img[x+1][y] - img[x-1][y]
            grad_y = img[x][y+1] - img[x][y-1]
            grad_xy = (grad_x ** 2 + grad_y ** 2) ** 0.5
            img_grad[x - 1][y - 1] = grad_xy
    return img_grad

def docanny(filename):
    SAVEDIR = 'plot/'
    READDIR = 'dataset/'
    img_gray = cv2.imread(READDIR + filename,cv2.IMREAD_GRAYSCALE)
    img_gauss= cv2.GaussianBlur(img_gray,(3,3),0)
    img_sobel = cv2.Sobel(img_gauss,cv2.CV_16S,1,0)
    img_sobel = cv2.convertScaleAbs(img_sobel)
    cv2.imshow(filename,img_sobel)
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




