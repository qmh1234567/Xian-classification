import random
import torch
import math
import numpy as np
from torchvision import transforms
from PIL import Image,ImageOps,ImageFilter

class Resize(object):
    def __init__(self,size,interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    
    def __call__(self,img):
        ratio = self.size[0]/self.size[1]
        w,h = img.size
        if w/h < ratio:
            t = int(h*ratio)
            w_padding = (t-w)//2
            img = img.crop((-w_padding,0,w+w_padding,h))
        else:
            t = int(w*ratio)
            h_padding = (t-h)//2
            img = img.crop((0,-h_padding,w,h+h_padding))
        img = img.resize(self.size,self.interpolation)
        return img

class RandomRotate(object):
    '''
    随机旋转图片
    '''
    def __init__(self,degree,p=0.5):
        self.degree = degree
        self.p = p
    
    def __call__(self,img):
        if random.random() < self.p:
            rotate_degree = random.uniform(-1*self.degree,self.degree)
            img = img.rotate(rotate_degree,Image.BILINEAR)
        return img
    

class RandomGaussianBlur(object):
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,img):
        if random.random() < self.p:
            # 高斯模糊是高斯低通滤波器 不保留细节   高斯滤波是高斯高通滤波器 保留细节
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return img

class Detail(object):
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,img):
        if random.random() < self.p:
            # 
            img = img.filter(ImageFilter.DETAIL)
        return img
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x  .size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixip_criterion(criterion, pred, y_a, y_b, lam):
    return lam *criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.3, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    # ratio = np.sqrt(1. - lam)
    cut_w = np.int(W * lam)
    cut_h = np.int(H * lam)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_train_transform(mean,std,size):
    train_transform = transforms.Compose([
        Resize((int(size*(256/224)),int(size*(256/224)))),
        # Detail(),
        transforms.CenterCrop(size),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        # transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std),
        # RandomErasing(mean=mean),
    ])
    return train_transform

def get_test_transform(mean,std,size):
    return transforms.Compose([
        Resize((int(size*(256/224)),int(size*(256/224)))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std),
    ])


def get_transforms(input_size=224,test_size=224,backbone=None):
    mean,std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if backbone is not None and backbone in ['pnasnet5large', 'nasnetamobile']:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    transformations = {}
    transformations['val_train'] = get_train_transform(mean,std,input_size)
    transformations['val_test'] = get_test_transform(mean,std,test_size)
    transformations['test'] = get_test_transform(mean,std,test_size)
    return transformations