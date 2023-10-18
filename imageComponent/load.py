import numpy as np
import math
import time
from collections import deque
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import sklearn.neighbors
import sklearn.metrics
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageFilter, ImageOps
import copy
from torchvision.models import AlexNet, EfficientNet, VGG, ResNet, VisionTransformer, SwinTransformer, ConvNeXt

import nvidia.dali as dali
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types

def CLaugment1(images, size, random_crop_area, strength):
    s = strength
    images = dali.fn.decoders.image_random_crop(images, random_area=random_crop_area, random_aspect_ratio=[min(1., 0.75 / s), max(1., 1.33 * s)],
                                                device='mixed', hw_decoder_load=0.75)
    images = dali.fn.resize(images, size=size)
    images = dali.fn.color_twist(images,
                                  brightness=1 + dali.fn.random.coin_flip(probability=0.8 * s) * dali.fn.random.uniform(range=(-0.4 * s, 0.4 * s)),
                                  contrast=1 + dali.fn.random.coin_flip(probability=0.8 * s) * dali.fn.random.uniform(range=(-0.4 * s, 0.4 * s)),
                                  saturation=1 + dali.fn.random.coin_flip(probability=0.8 * s) * dali.fn.random.uniform(range=(-0.2 * s, 0.2 * s)),
                                  hue=dali.fn.random.coin_flip(probability=0.8 * s) * dali.fn.random.uniform(range=(-36 * s, 36 * s)))
    images = dali.fn.saturation(images, saturation=dali.fn.random.coin_flip(probability=1 - 0.2 * s) * 1.)
    images = dali.fn.flip(images, horizontal=dali.fn.random.coin_flip(probability=0.5 * s))
    images = dali.fn.gaussian_blur(images, sigma=dali.fn.random.uniform(range=(0.1 * s, 2.0 * s)))
    images = dali.fn.transpose(images, perm=[2, 0, 1])
    return images

def CLaugment2(images, size, random_crop_area, strength):
    s = strength
    images = dali.fn.decoders.image_random_crop(images, random_area=random_crop_area, random_aspect_ratio=[min(1., 0.75 / s), max(1., 1.33 * s)],
                                                device='mixed', hw_decoder_load=0.75)
    images = dali.fn.resize(images, size=size)
    images = dali.fn.color_twist(images,
                                  brightness=1 + dali.fn.random.coin_flip(probability=0.8 * s) * dali.fn.random.uniform(range=(-0.4 * s, 0.4 * s)),
                                  contrast=1 + dali.fn.random.coin_flip(probability=0.8 * s) * dali.fn.random.uniform(range=(-0.4 * s, 0.4 * s)),
                                  saturation=1 + dali.fn.random.coin_flip(probability=0.8 * s) * dali.fn.random.uniform(range=(-0.2 * s, 0.2 * s)),
                                  hue=dali.fn.random.coin_flip(probability=0.8 * s) * dali.fn.random.uniform(range=(-36 * s, 36 * s)))
    images = dali.fn.saturation(images, saturation=dali.fn.random.coin_flip(probability=1 - 0.2 * s) * 1.)
    images = dali.fn.flip(images, horizontal=dali.fn.random.coin_flip(probability=0.5 * s))
    doblur = dali.fn.random.coin_flip(probability=0.1 * s, dtype=types.DALIDataType.BOOL)
    gimages = dali.fn.gaussian_blur(images, sigma=dali.fn.random.uniform(range=(0.1 * s, 2.0 * s)))
    images = doblur * gimages + (doblur ^ True) * images
    dosolarize = dali.fn.random.coin_flip(probability=0.2 * s, dtype=types.DALIDataType.BOOL)
    iimages = types.Constant(255, dtype=types.DALIDataType.UINT8) - images
    mthreshold = images > 128
    simages = iimages * mthreshold + (mthreshold ^ True) * images
    images = dosolarize * simages + (dosolarize ^ True) * images
    images = dali.fn.transpose(images, perm=[2, 0, 1])
    return images

@dali.pipeline_def
def DALICLImageFolders(root, size, random_crop_area=[0.1, 1.0], strength=1, initial_fill=50000):
    mean = dali.types.Constant([0.485 * 255, 0.456 * 255, 0.406 * 255])[:, dali.newaxis, dali.newaxis]
    std = dali.types.Constant([0.229 * 255, 0.224 * 255, 0.225 * 255])[:, dali.newaxis, dali.newaxis]
    
    images, labels = dali.fn.readers.file(file_root=root, name='reader', random_shuffle=True, initial_fill=initial_fill)
    images1, images2 = CLaugment1(images, size, random_crop_area, strength), CLaugment2(images, size, random_crop_area, strength)
    images1 = dali.fn.normalize(images1, axes=(1, 2), mean=mean, stddev=std)
    images2 = dali.fn.normalize(images2, axes=(1, 2), mean=mean, stddev=std)
    labels = dali.fn.squeeze(labels, axes=0)
    return images1, images2, labels

@dali.pipeline_def
def DALIHashCLImageFolders(files, size, random_crop_area=[0.1, 1.0], initial_fill=50000):
    mean = dali.types.Constant([0.485 * 255, 0.456 * 255, 0.406 * 255])[:, dali.newaxis, dali.newaxis]
    std = dali.types.Constant([0.229 * 255, 0.224 * 255, 0.225 * 255])[:, dali.newaxis, dali.newaxis]
    
    images, labels = dali.fn.readers.file(files=files, name='reader', random_shuffle=True, initial_fill=initial_fill)
    images1, images2 = CLaugment1(images, size=size, random_crop_area=random_crop_area), CLaugment2(images, size=size, random_crop_area=random_crop_area)
    images1 = dali.fn.normalize(images1, axes=(1, 2), mean=mean, stddev=std)
    images2 = dali.fn.normalize(images2, axes=(1, 2), mean=mean, stddev=std)
    labels = dali.fn.squeeze(labels, axes=0)
    return images1, images2, labels

@dali.pipeline_def
def DALISupervisedImageFolders(root, size, random_crop_area=[0.1, 1.0], initial_fill=50000):
    mean = dali.types.Constant([0.485 * 255, 0.456 * 255, 0.406 * 255])[:, dali.newaxis, dali.newaxis]
    std = dali.types.Constant([0.229 * 255, 0.224 * 255, 0.225 * 255])[:, dali.newaxis, dali.newaxis]
    
    images, labels = dali.fn.readers.file(file_root=root, name='reader', random_shuffle=True, initial_fill=initial_fill)
    
    images = dali.fn.decoders.image_random_crop(images, random_area=random_crop_area, random_aspect_ratio=[0.75, 4 / 3],
                                                device='mixed', hw_decoder_load=0.75)
    images = dali.fn.resize(images, size=size)
    
    images = dali.fn.flip(images, horizontal=dali.fn.random.coin_flip(probability=0.5))
    images = dali.fn.transpose(images, perm=[2, 0, 1])
    images = dali.fn.normalize(images, axes=(1, 2), mean=mean, stddev=std)
    labels = dali.fn.squeeze(labels, axes=0)
    return images, labels

@dali.pipeline_def
def DALIValdImageFolders(root, size):
    mean = dali.types.Constant([0.485 * 255, 0.456 * 255, 0.406 * 255])[:, dali.newaxis, dali.newaxis]
    std = dali.types.Constant([0.229 * 255, 0.224 * 255, 0.225 * 255])[:, dali.newaxis, dali.newaxis]
    
    images, labels = dali.fn.readers.file(file_root=root, name='reader', random_shuffle=True)
    images = dali.fn.decoders.image(images, device='mixed', hw_decoder_load=0.75)
    images = dali.fn.resize(images, size=size, mode='not_smaller')
    images = dali.fn.crop(images, crop=size)
    
    images = dali.fn.transpose(images, perm=[2, 0, 1])
    images = dali.fn.normalize(images, axes=(1, 2), mean=mean, stddev=std)
    labels = dali.fn.squeeze(labels, axes=0)
    return images, labels

@dali.pipeline_def
def DALIHadhValdImageFolders(root, files, size):
    mean = dali.types.Constant([0.485 * 255, 0.456 * 255, 0.406 * 255])[:, dali.newaxis, dali.newaxis]
    std = dali.types.Constant([0.229 * 255, 0.224 * 255, 0.225 * 255])[:, dali.newaxis, dali.newaxis]
    
    images, labels = dali.fn.readers.file(file_root=root, files=files, name='reader', random_shuffle=False)
    images = dali.fn.decoders.image(images, device='mixed', hw_decoder_load=0.75)
    images = dali.fn.resize(images, size=size, mode='not_smaller')
    images = dali.fn.crop(images, crop=size)
    
    images = dali.fn.transpose(images, perm=[2, 0, 1])
    images = dali.fn.normalize(images, axes=(1, 2), mean=mean, stddev=std)
    labels = dali.fn.squeeze(labels, axes=0)
    return images, labels

@dali.pipeline_def
def DALIOverlapCLImageFolders(files, size, random_crop_area=[0.2, 1.0]):
    mean = dali.types.Constant([0.485 * 255, 0.456 * 255, 0.406 * 255])[:, dali.newaxis, dali.newaxis]
    std = dali.types.Constant([0.229 * 255, 0.224 * 255, 0.225 * 255])[:, dali.newaxis, dali.newaxis]
    
    images, labels = dali.fn.readers.file(files=files, name='reader')
    images1, images2 = CLaugment1(images, size, random_crop_area, 1), CLaugment2(images, size, random_crop_area, 1)
    images1 = dali.fn.normalize(images1, axes=(1, 2), mean=mean, stddev=std)
    images2 = dali.fn.normalize(images2, axes=(1, 2), mean=mean, stddev=std)
    labels = dali.fn.squeeze(labels, axes=0)
    return images1, images2, labels