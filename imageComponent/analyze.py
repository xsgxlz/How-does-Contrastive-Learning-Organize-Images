import numpy as np
import math
import time
import collections
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

class Normalize(nn.Module):
    def __init__(self, p=2.0, dim=1, eps=1e-12, out=None):
        super().__init__()
        self.p = p
        self.dim = dim
        self.eps = eps
        self.out = out
    def forward(self, x):
        return F.normalize(x, self.p, self.dim, self.eps, self.out)

def feature_eval(model, embedding_dim, trainloader, rawtrainloader, valloader, num_class, batch_size, tolerate, device,
                 init_probe=None, normalize=False, use_amp=True, verbose=False):
    scaler = T.cuda.amp.GradScaler(enabled=use_amp)
    if not init_probe:
        probe = nn.Linear(embedding_dim, num_class)
    else:
        probe = init_probe
    probe = probe.to(device, memory_format=T.channels_last).train()
    with T.no_grad():
        probe.bias *= 0
    optim = T.optim.AdamW(probe.parameters(), lr=2e-3 / 64 * batch_size)
    #scheduler = T.optim.lr_scheduler.LambdaLR(optim, WarmupCosineAnnealingWarmRestarts(0.3, 5, 10, 1e-2))
    best_train_acc = 0
    best_raw_train_acc = 0
    best_val_acc = 0
    t = 0
    while t < tolerate:
        model = model.eval().to(device)
        probe = probe.train()
        label_record = {'gt':[], 'pred':[]}
        tm = time.time()
        t += 1
        train_acc = 0
        for data in trainloader:
            with T.no_grad():
                with T.autocast(device_type='cuda', dtype=T.float16, enabled=use_amp):
                    x = data[0]['image']
                    y = data[0]['label'].long()
                    label_record['gt'].append(y.detach().cpu().numpy())
                    x, y = x.to(device, memory_format=T.channels_last), y.to(device)
                    feature = model(x)
                    if normalize:
                        feature = F.normalize(feature)
                    else:
                        feature = feature.float()
            pred = probe(feature)
            label_record['pred'].append(T.argmax(pred, dim=1).detach().cpu().numpy())
            loss = F.cross_entropy(pred, y, label_smoothing=0.1)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
        label_record['gt'] = np.concatenate(label_record['gt'], axis = 0)
        label_record['pred'] = np.concatenate(label_record['pred'], axis = 0)
        
        train_acc = sklearn.metrics.accuracy_score(label_record['gt'], label_record['pred'])
        full_classifier = nn.Sequential(model, Normalize(), probe) if normalize else nn.Sequential(model, probe)
        raw_train_acc = val_eval(full_classifier, rawtrainloader, device, use_amp=use_amp)
        val_acc = val_eval(full_classifier, valloader, device, use_amp=use_amp)
        if val_acc > best_val_acc:
            t = 0
        best_train_acc = max(train_acc, best_train_acc)
        best_raw_train_acc = max(raw_train_acc, best_raw_train_acc)
        best_val_acc = max(val_acc, best_val_acc)
        if verbose:
            print(train_acc, raw_train_acc, val_acc)
            print(best_train_acc, best_raw_train_acc, best_val_acc)
            print(time.time() - tm, end='\n\n')
    return (best_train_acc, best_raw_train_acc, best_val_acc), probe

def val_eval(model, valloader, device, use_amp=True):
    model.eval().to(device)
    label_record = {'gt':[], 'pred':[]}
    with T.inference_mode():
        with T.autocast(device_type='cuda', dtype=T.float16, enabled=use_amp):
            for data in valloader:
                x = data[0]['image']
                y = data[0]['label'].long()
                label_record['gt'].append(y.detach().cpu().numpy())
                x, y = x.to(device, memory_format=T.channels_last), y.to(device)
                pred = model(x)
                label_record['pred'].append(T.argmax(pred, dim=1).detach().cpu().numpy())
    label_record['gt'] = np.concatenate(label_record['gt'], axis = 0)
    label_record['pred'] = np.concatenate(label_record['pred'], axis = 0)
    acc = sklearn.metrics.accuracy_score(label_record['gt'], label_record['pred'])
    return acc

#@T.jit.script
def contrastive_loss(q, k, temperature: float=0.2, normalized: bool=False):
    if not normalized:
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
    sim = q @ k.T / temperature
    labels = T.arange(len(q), device=q.device)
    return F.cross_entropy(sim, labels) * (2 * temperature)

def extract_feature(loader, model, device, normalize=False, use_amp=True):
    model.eval()
    with T.no_grad():
        features, label = [], []
        for data in loader:
            with T.autocast(device_type='cuda', dtype=T.float16, enabled=use_amp):
                #print(data[0]['image'])
                x = data[0]['image'].to(device, memory_format=T.channels_last)
                y = data[0]['label']
                feat = model(x)
                if normalize:
                    feat = F.normalize(feat)
                features.append(feat.cpu())
                label.append(y.cpu())
            x = T.cat(features, dim=0).to(device)
            y = T.cat(label, dim=0).to(device).long()
    return x, y

def to_spare(adj, k):
    kidx = len(adj) - k + 1
    adj = (adj >= T.kthvalue(adj, k=kidx, dim=1)[0][:, None]) * adj
    adj = (adj + adj.T) / 2
    return adj

def cos_similarity(x, y=None):
    device = x.device
    x = F.normalize(x)
    if y == None:
        flag = True
        y = x
    else:
        flag = False
        y = F.normalize(y)
    similarity = x @ y.T
    if flag:
        idx = T.arange(len(x), device=device)
        similarity[idx, idx] = 1
    return similarity

def modularity(adj, communities, spare=None):
    if spare:
        adj = to_spare(adj, spare)
    m = adj.sum() / 2
    k = adj.sum(dim=1)
    delta = communities[:, None] == communities[None]
    Q = ((adj - k[:, None] @ k[None] / (2 * m)) * delta).sum() / (m * 2)
    return Q

def calinski_harabasz_score(x, y):
    device = x.device
    nE = len(x)
    k = y.max() + 1
    data_center = x.mean(dim = 0)
    nq = []
    cluster_centers = []
    for i in range(k):
        in_cluster = (y == i)
        cluster_centers.append(x[in_cluster].mean(dim=0))
        nq.append(in_cluster.sum())
    cluster_centers = T.stack(cluster_centers, dim=0)
    nq = T.stack(nq, dim=0).to(device)
    
    intra_disp = (x - cluster_centers[y]).square().sum()
    extra_disp = (nq * ((cluster_centers - data_center[None]).square()).sum(dim=1)).sum()
    return extra_disp / intra_disp / (k - 1) * (nE - k)

#@T.jit.script
def modularity_inplace(adj, communities, spare=None):
    if isinstance(spare, int):
        adj = to_spare(adj, spare)
    m = adj.sum() / 2
    k = adj.sum(dim=1)
    kk = k[:, None] * k[None]
    kk /= (2 * m)
    adj -= kk
    delta = communities[:, None] == communities[None]
    adj *= delta
    Q = adj.sum() / (2 * m)
    del m, k, kk, adj, delta
    return Q

#@T.jit.script
def _edge_weight(sim_matrix, temperature: float):
    n = sim_matrix.shape[0]
    sim_matrix /= temperature
    sim_matrix = sim_matrix.exp()
    row_sum = sim_matrix.sum(dim=1, keepdim=True)
    col_sum = sim_matrix.sum(dim=0, keepdim=True)
    sim_matrix *= sim_matrix
    sim_matrix *= n * n
    sim_matrix /= row_sum
    sim_matrix /= col_sum
    return sim_matrix

#@T.jit.script
def pnorm_sim_matrix(X, diag_ninf: bool=True):
    device = X.device
    similarity = F.pdist(X) / math.sqrt(X.shape[1])
    similarity /= -similarity.mean()
    similarity = similarity.cpu()
    sim_matrix = T.zeros((len(X), len(X)))#, device=device)
    idx = T.triu_indices(len(X), len(X), offset=1)#, device=device)
    sim_matrix[idx[0], idx[1]] = similarity
    sim_matrix[idx[1], idx[0]] = similarity
    if diag_ninf:
        idx = T.arange(len(X))
        sim_matrix[idx, idx] = float('-inf')
    del similarity, idx
    return sim_matrix.to(device)

def relative_local_density_score(X, y, temperature: float=1):
    device = X.device
    sim_matrix = pnorm_sim_matrix(X)
    if isinstance(temperature, collections.abc.Iterable):
        sim_matrix.cpu()
        scores = []
        for i in temperature:
            weight = _edge_weight(sim_matrix.clone().to(device), i)
            scores.append(modularity_inplace(weight, y))
            del weight
        return scores
    else:
        weight = _edge_weight(sim_matrix, temperature)
        return modularity_inplace(weight, y)

def sim_graph(X, temperature):
    sim_matrix = pnorm_sim_matrix(X)
    weight = _edge_weight(sim_matrix, temperature)
    return weight

def standardize(x):
    return (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True)

def take_class(x, y, num_class):
    idx = y < num_class
    return x[idx], y[idx]