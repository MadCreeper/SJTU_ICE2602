import cv2
from cv2 import cv2 # make vscode not complain
from matplotlib import pyplot as plt
import numpy as np

def plotGRAY(filename):
    SAVEDIR = 'plot/'
    READDIR = 'images/'
    img_gray = cv2.imread(READDIR + filename,cv2.IMREAD_GRAYSCALE)
    
    data = img_gray.ravel().astype(int)
    bins = np.array(range(256))  - 0.01  # 防止值落在边界上
    plt.hist(data,bins=bins,density=True,rwidth=1)
    
    plt.title('GRAY Histogtam of ' + filename)
    plt.savefig(SAVEDIR + filename +'_GRAY_Hist.png')
    #plt.show()
    plt.close()

if __name__ == '__main__':
    files = [
    'img1.jpg',
    'img2.jpg',
    'img3.jpg'
    ] 
    for file in files:
        plotGRAY(file)  




