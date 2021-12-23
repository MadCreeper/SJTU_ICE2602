# SJTU EE208

import time
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from cv2 import (imshow, imread, waitKey, destroyAllWindows)

normalized = lambda x: x / np.linalg.norm(x)


print('Load model: ResNet50')
model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
#model = torchvision.models.resnet50(pretrained=True)
#print(model)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])



def features(x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    
    return x

def normalized(vec):
    return vec / np.linalg.norm(vec)

def calc_similarity(feature1, feature2):
    return np.dot(normalized(feature1), normalized(feature2))

def load_img_file(filename):
    target_image = default_loader(filename)
    target_image = trans(target_image)
    target_image = torch.unsqueeze(target_image, 0)
    return target_image


if __name__ == "__main__":
    imgs_path = "./imgs2"
    target_path = 'cat2.jpg'
    
    start = time.time()

    target_image = load_img_file(target_path)
    target_feature = features(target_image)
    target_feature = target_feature.detach().numpy().reshape(-1)
    print(f"Image search target {target_path}")

    similarities = []
    print("Started extracting features in testset ...")
    for root, dirs, files in os.walk(imgs_path, topdown=True):
        for name in files:
            filename = os.path.join(root, name)
            if not filename.endswith('.jpg') or filename.endswith('.png'):
                continue
            #print(filename)
            img = load_img_file(filename)
            feature = features(img).detach().numpy().reshape(-1)
            sim = calc_similarity(feature, target_feature)
            #print(sim)
            similarities.append((filename, sim))

    print('Done! Time for extracting features: {:.2f}'.format(time.time() - start))

similarities = sorted(similarities,reverse = True, key = lambda x: x[1])
#print(similarities)

top_cnt = 5
print(f"Showing top {top_cnt} closest matches: ")
for filename, sim_score in reversed(similarities[:top_cnt]):
    filename = os.path.normpath(os.path.join('./', filename))
    print(filename)
    img = imread(filename)
 
    imshow("Similarity: " + str(sim_score), img)

waitKey(0)
destroyAllWindows()
#print(image_feature.ravel(), image_feature2.ravel())
#print(image_feature.shape, image_feature2.shape)
 #   print(np.dot(normalized(image_feature), normalized(image_feature2)))
#print(np.dot(normalized(image_feature.ravel()) ,normalized(image_feature2.ravel())))
#print(np.linalg.norm(normalized(image_feature.ravel()) - normalized(image_feature2.ravel())))


#print('Save features!')
#np.save('features.npy', image_feature)
    