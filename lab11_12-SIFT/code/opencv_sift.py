# https://www.pythonheidong.com/blog/article/301412/de52ba77b3331350ccb7/
import cv2
import numpy as np


def sift_kp(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    kp_image = cv2.drawKeypoints(gray_image, kp, None)
    return kp_image, kp, des


def get_good_match(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)  # des1为模板图，des2为匹配图
    matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


img1 = cv2.imread(r'target.jpg')
img2 = cv2.imread(r'./dataset/6.jpg')

kpimg1, kp1, des1 = sift_kp(img1)
kpimg2, kp2, des2 = sift_kp(img2)

#cv2.imshow('img1',np.hstack((img1,kpimg1)))
#cv2.waitKey(0)
#cv2.imshow('img2',np.hstack((img2,kpimg2)))
#cv2.waitKey(0)

goodMatch = get_good_match(des1, des2)
all_goodmatch_img= cv2.drawMatches(img1, kp1, img2, kp2, goodMatch, None, flags=2)
# goodmatch_img自己设置前多少个goodMatch[:10]
goodmatch_img = cv2.drawMatches(img1, kp1, img2, kp2, goodMatch[:50], None, flags=2)

#cv2.imshow('all_goodmatch_img', all_goodmatch_img)
#cv2.waitKey(0)
cv2.imshow('goodmatch_img', goodmatch_img)
cv2.imwrite('match3.png',goodmatch_img)
cv2.waitKey(0)