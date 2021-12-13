# SJTU EE208

import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader

print('Load model: ResNet50')
model = torch.hub.load('pytorch/vision', 'alexnet', pretrained=True)
#model = torchvision.models.resnet50(pretrained=True)
print(model)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

print('Prepare image data!')
test_image = default_loader('dog.jpg')
input_image = trans(test_image)
input_image = torch.unsqueeze(input_image, 0)

test_image2 = default_loader('cat.jpg')
input_image2 = trans(test_image2)
input_image2 = torch.unsqueeze(input_image2, 0)

def features(x):
    x = model.conv2(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.conv2d(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.conv2d(x)
    x = model.relu(x)
    x = model.conv2d(x)
    x = model.relu(x)
    x = model.conv2d(x)
    x = model.relu(x)
   
    #x = model.avgpool(x)
    
    return x

def normalized(vec):
    return vec / np.linalg.norm(vec)

print('Extract features!')
start = time.time()
image_feature = features(input_image)
image_feature = image_feature.detach().numpy()
#print(image_feature)
#print(model(input_image).detach().numpy())
image_feature2 = features(input_image2).detach().numpy()
#print(image_feature.ravel(), image_feature2.ravel())
print(image_feature.shape, image_feature2.shape)

print(np.dot(normalized(image_feature.ravel()) ,normalized(image_feature2.ravel())))
#print(np.linalg.norm(normalized(image_feature.ravel()) - normalized(image_feature2.ravel())))
print('Time for extracting features: {:.2f}'.format(time.time() - start))


print('Save features!')
np.save('features.npy', image_feature)
