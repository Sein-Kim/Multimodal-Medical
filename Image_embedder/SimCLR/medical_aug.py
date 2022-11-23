import argparse
import torch
from torchvision import models

from torchvision.transforms import transforms
from torchvision import transforms, datasets

def medical_aug(img_t, size=96, s =1):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            transforms.ToTensor()])
    trans = transforms.ToPILImage()
    img = trans(img_t)
    aug_img = data_transforms(img)
    return aug_img