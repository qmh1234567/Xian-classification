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

def mixup_data(args,x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda(args.gpu)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length,p=0.3):
        self.n_holes = n_holes
        self.length = length
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if random.random() < self.p:
            h = img.size(1)
            w = img.size(2)

            mask = np.ones((h, w), np.float32)

            for n in range(self.n_holes):
                # (x,y)表示方形补丁的中心位置
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                mask[y1: y2, x1: x2] = 0.

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask
        return img


def get_train_transform(mean,std,size):
    train_transform = transforms.Compose([
        Resize((int(size*(256/224)),int(size*(256/224)))),
        transforms.CenterCrop(size),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std),
        # Cutout(n_holes=1, length=16),
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