import torch as T
import torch.nn.functional as F
import torch.nn as nn
import sklearn.metrics

import random
from itertools import chain

from .analyze import sim_graph
from .GCN import GCNWithLoop, GCNSequential, BN_SELU

class NNHead():
    def __init__(self, NN_func, tolerate, label_smoothing, lr, use_amp=False, verbose=0):
        self.NN_func = NN_func
        self.tolerate = tolerate
        self.label_smoothing = label_smoothing
        self.lr = lr
        self.use_amp = use_amp
        self.verbose = verbose
    
    def fit(self, X, y, tX, ty):
        self.dim = X.shape[1]
        self.num_class = y.max().item() + 1
        self.device = X.device
        self.NN = self.NN_func(self.dim, self.num_class, device=self.device)
        optim = T.optim.AdamW(self.NN.parameters(), lr=self.lr)
        scaler = T.cuda.amp.GradScaler(enabled = self.use_amp)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        best_acc = 0
        best_val_acc = 0
        t = 0
        total_epoch = 0
        while t < self.tolerate:
            with T.autocast(device_type = 'cuda', dtype = T.float16, enabled = self.use_amp):
                pred = self.NN(X)
                pred_label = T.argmax(pred, dim=1).detach().cpu().numpy()
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none = True)
            
            t += 1
            total_epoch += 1
            acc = sklearn.metrics.accuracy_score(pred_label, y.cpu().numpy())
            val_acc = self.score(tX, ty)
            best_val_acc = max(best_val_acc, val_acc)
            if self.verbose > 0:
                print('epoch:%d\tacc:%f\tvacc:%f' %(total_epoch, acc, best_val_acc))
            if acc > best_acc:
                best_acc = acc
                t = 0
        self.best_acc = best_acc
        self.best_val_acc = best_val_acc
        return self
    
    def score(self, X, y):
        with T.autocast(device_type = 'cuda', dtype = T.float16, enabled = self.use_amp):
            with T.inference_mode():
                pred = self.NN(X)
                pred_label = T.argmax(pred, dim=1).detach().cpu().numpy()
                acc = sklearn.metrics.accuracy_score(pred_label, y.cpu().numpy())
        return acc
    
class GraphHead():
    def __init__(self, temperature, mask_p, tolerate, label_smoothing, lr, use_amp=False, verbose=0):
        self.temperature = temperature
        self.mask_p = mask_p
        self.tolerate = tolerate
        self.label_smoothing = label_smoothing
        self.lr = lr
        self.use_amp = use_amp
        self.verbose = verbose
    def fit(self, tX, ty, vX, vy):
        device = tX.device
        self.num_class = (ty.max() + 1).item()
        self.len_t = len(tX)
        self.dim = tX.shape[1]
        with T.no_grad():
            self.node_feat = T.cat([tX, vX], dim=0)
            self.node_label = T.cat([ty, vy], dim=0)

            self.adj = sim_graph(self.node_feat, self.temperature).to(device)

            #self.node_feat = T.cat([self.node_feat, label_feat], dim=1)
        self.embedding = nn.Embedding(self.num_class + 1, 16, device=device)
        self.GCN = GCNSequential(GCNWithLoop(16, 16, activation=BN_SELU(16), device=device),
                                 #GCNWithLoop(16, 16, activation=BN_SELU(16), device=device),
                                 GCNWithLoop(16, self.num_class, activation=None, device=device))
        
        optim = T.optim.AdamW(chain(self.embedding.parameters(), self.GCN.parameters()), lr=self.lr)
        scaler = T.cuda.amp.GradScaler(enabled = self.use_amp)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        best_acc = 0
        best_val_acc = 0
        t = 0
        total_epoch = 0
        while t < self.tolerate:
            with T.autocast(device_type = 'cuda', dtype = T.float16, enabled = self.use_amp):
                p = random.uniform(*self.mask_p)
                mask = T.distributions.bernoulli.Bernoulli(p).sample((self.len_t, )).bool().to(device)
                mask0 = T.cat([mask, T.zeros(len(vX), dtype=T.bool, device=device)])
                mask1 = T.cat([mask, T.ones(len(vX), dtype=T.bool, device=device)])
                masked_feat = self.node_label.clone()
                masked_feat[mask1] = self.num_class
                
                pred = self.GCN(self.adj, self.embedding(masked_feat))
                pred_label = T.argmax(pred[mask0], dim=1).detach().cpu().numpy()
                acc = sklearn.metrics.accuracy_score(pred_label, self.node_label[mask0].cpu().numpy())
                loss = loss_fn(pred[mask0], self.node_label[mask0])
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none = True)
            
            t += 1
            total_epoch += 1
            
            val_acc = self.score()
            best_val_acc = max(best_val_acc, val_acc)
            if self.verbose > 0:
                print('epoch:%d\tacc:%f\tvacc:%f' %(total_epoch, acc, best_val_acc))
            if acc > best_acc:
                best_acc = acc
                t = 0
        self.best_acc = best_acc
        self.best_val_acc = best_val_acc
        return self
    
    def score(self, mode='val', p=None):            
        with T.autocast(device_type = 'cuda', dtype = T.float16, enabled = self.use_amp):
            with T.inference_mode():
                masked_feat = self.node_label.clone()
                masked_feat[self.len_t:] = self.num_class
                pred = self.GCN(self.adj, self.embedding(masked_feat))
                
                pred_label = T.argmax(pred[self.len_t:], dim=1).detach().cpu().numpy()
                acc = sklearn.metrics.accuracy_score(pred_label, self.node_label[self.len_t:].cpu().numpy())
        return acc