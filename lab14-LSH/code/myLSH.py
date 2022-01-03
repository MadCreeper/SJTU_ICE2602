import time
import os
import math
import numpy as np
import cv2
import colorHist

def calc_similarity(feature1, feature2):
    def normalized(vec): # 标准化(单位)向量
        if np.linalg.norm(vec) == 0:
            return 0
        return vec / np.linalg.norm(vec)
    return np.dot(normalized(feature1), normalized(feature2))

def hamming(p, d, C):
    hcode = []
    for p_i in p:
        hcode += [1] * int(p_i)
        hcode += [0] * int(C - p_i)
    return hcode

# bitwise projection 
def projection(set1, set2): 
    proj = [set1[x] for x in set2]
    return proj

def lsh(img, proj_set):
    C = 2
    H, W = img.shape[0], img.shape[1]
    #print(H,W)
    quadrants = [((0,0),(H//2, W//2)) , ((0, W//2), (H//2, W)),
                 ((H//2,0), (H, W//2)), ((H//2, W//2), (H, W))]    
    # divide into 4 sections
    feat = []
    for sect in quadrants:
        c1, c2 = sect
        #print(c1, c2)
        sub_img = img[c1[0]:c2[0], c1[1]:c2[1]] 
        rgb = colorHist.RGB_hist(sub_img)
        
        rgbf = np.digitize(rgb, [0, 0.3, 0.6]) - 1     # 按照0~0.3, 0.3~0.6, 0.6~1 分类
        #print(rgbf)
        feat += list(rgbf)
        #print(rgb)
    
    #print(feature)
    hcode = hamming(feat, len(feat), C)
    #print(hcode)
    hashed = projection(hcode, proj_set)
    return hashed

if __name__ == "__main__":
    imgs_path = "./Dataset"
    target_path = 'target.jpg'
    proj_set = [4,8,12,16]
    target_name = target_path[target_path.find('/'):]
    
    start = time.time()

    target_image = img = cv2.imread(target_path)
    target_feature = lsh(target_image, proj_set=proj_set)
    #print(target_feature)
    print(target_feature)
    
    print(f"Image search target {target_path}")

    similarities = []
    print("Started extracting features in testset ...")
    for root, dirs, files in os.walk(imgs_path, topdown=True):
        for name in files:
            filename = os.path.join(root, name)
            if not filename.endswith('.jpg') or filename.endswith('.png'):
                continue
            #print(filename)
            img = cv2.imread(filename)
            feature = lsh(img, proj_set=proj_set)
            #print(feature)
            sim = calc_similarity(feature, target_feature)
            
            #print(sim)
            similarities.append((filename, sim))

    print('Done! Time for extracting features: {:.2f}'.format(time.time() - start))
    
    similarities = sorted(similarities, reverse = True, key = lambda x: x[1])
    #print(similarities)

    top_cnt = 5
    print(f"Showing top {top_cnt} closest matches: ")
    for i, (filename, sim_score) in enumerate((similarities[:top_cnt])):
        filename = os.path.normpath(os.path.join('./', filename))
        print(f"Top{top_cnt - i}", filename)
        img = cv2.imread(filename)
        print(sim_score)
        cv2.imshow(f"#{top_cnt - i}_sim=" + str(round(sim_score,3)), img)
        cv2.imwrite(f"match_top_{top_cnt - i}_{str(proj_set)}.png", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
