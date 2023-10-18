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

from .load import *
from .analyze import *

def resnetToCifar(resnet):
    resnet.conv1 = nn.Conv2d(3, 64, (3, 3), padding='same')
    resnet.maxpool = nn.Identity()
    return resnet

def ResnetCifar(num_layer, target_dim=512):
    model = resnetToCifar(eval('torchvision.models.resnet%d()' %num_layer))
    model.fc = nn.Linear(model.fc.in_features, target_dim)
    return model

def ViTCifar(target_dim=512):
    model = VisionTransformer(image_size=32, patch_size=4, num_layers=16, num_heads=8, hidden_dim=512, mlp_dim=1536,
                          dropout=0.2, attention_dropout=0.2)
    model.heads = nn.Linear(512, target_dim)
    return model

def build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    mlp = []
    mlp.append(nn.BatchNorm1d(input_dim))
    mlp.append(nn.SELU(inplace = True))
    mlp.append(nn.Linear(input_dim, mlp_dim, bias=False))
    mlp += [nn.BatchNorm1d(mlp_dim), nn.SELU(inplace = True), nn.Linear(mlp_dim, mlp_dim, bias=False)] * (num_layers - 2)
    mlp.append(nn.BatchNorm1d(mlp_dim))
    mlp.append(nn.SELU(inplace = True))
    mlp.append(nn.Linear(mlp_dim, output_dim))
    if last_bn:
        mlp.append(nn.BatchNorm1d(output_dim, affine = False))
    return nn.Sequential(*mlp)

class MoCo(nn.Module):
    def __init__(self, base_encoder, deque_len, mlp_dim=None, m=0.99):
        super(MoCo, self).__init__()
        self.m = m
        self.deque_len = deque_len
        self.memory_queue1 = deque(maxlen=deque_len)
        self.memory_queue2 = deque(maxlen=deque_len)
        self.base_encoder = base_encoder
        self._build_predictor(mlp_dim)
        self.momentum_encoder = copy.deepcopy(base_encoder)
        for param_m in self.momentum_encoder.parameters():
            param_m.requires_grad = False  # not update by gradient

    def _build_predictor(self, mlp_dim):
        if isinstance(self.base_encoder, ResNet):
            target_dim = self.base_encoder.fc.out_features
        if isinstance(self.base_encoder, VisionTransformer):
            target_dim = self.base_encoder.heads.out_features
        if not mlp_dim:
            mlp_dim = target_dim
        self.predictor = build_mlp(3, target_dim, mlp_dim, target_dim)
        self.target_dim = target_dim

    @T.no_grad()
    def _update_momentum_encoder(self):
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * self.m + param_b.data * (1 - self.m)

    def forward(self, x1, x2, y=None, additional_loss=None, alpha=None):
        with T.no_grad():
            self._update_momentum_encoder()
            batch_k1 = F.normalize(self.momentum_encoder(x1), dim=1)
            batch_k2 = F.normalize(self.momentum_encoder(x2), dim=1)
            
            self.memory_queue1.appendleft(batch_k1)
            self.memory_queue2.appendleft(batch_k2)
            k1 = T.cat(list(self.memory_queue1))
            k2 = T.cat(list(self.memory_queue2))
            
        q1 = F.normalize(self.predictor(self.base_encoder(x1)), dim=1)
        q2 = F.normalize(self.predictor(self.base_encoder(x2)), dim=1)
        
        closs = contrastive_loss(q1, k2, normalized=True) + contrastive_loss(q2, k1, normalized=True)
        
        if additional_loss == 'CH':
            CH = calinski_harabasz_score(T.cat([q1, q2]), T.cat([y, y]))
            return alpha * CH + closs, (alpha * CH).item(), closs.item()

        return closs