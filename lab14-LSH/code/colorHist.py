import cv2
from cv2 import cv2 # make vscode not complain
from matplotlib import pyplot as plt
import numpy as np
def countRGB(img):
    (B, G, R) = cv2.split(img)
    return [np.sum(R), np.sum(G), np.sum(B)]

def RGB_hist(img_bgr):
    #SAVEDIR = 'plot/'
    #READDIR = 'images/'
    #img_bgr = cv2.imread(READDIR + filename)
    #img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    RGB = countRGB(img_bgr)
    RGB_normalized = RGB / np.sum(RGB)
    return RGB_normalized
    """
    #name_list = ['R','G','B']
    #X = [1,2,3]
    Y = 
    plt.bar(X,Y ,color=['r','g','b'],tick_label=name_list)
    for a,b in zip(X,Y):  
        plt.text(a, b, round(b,3) , ha='center', va= 'bottom',fontsize=11)  
        plt.title('RGB Bar graph of ' + filename)
    #plt.show()
    plt.savefig(SAVEDIR + filename +'_RGB_Bar.png')
    plt.close()
    """
    
if __name__ == '__main__':
    files = [
    'img1.jpg',
    'img2.jpg',
    'img3.jpg'
    ] 
    for file in files:
        img = cv2.imread(file)
        print(RGB_hist(img))  




