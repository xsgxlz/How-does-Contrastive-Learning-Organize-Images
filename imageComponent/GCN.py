import torch as th
from torch import nn
from torch.nn import init

from .analyze import *

#from dgl

class GCNWithLoop(nn.Module):
    def __init__(self, in_feats, out_feats, norm="both", bias=True, activation=None, device=None):
        super().__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        self.self_loop = nn.Parameter(th.Tensor(1))
        self.feat = None
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self._activation = activation
        self.to(device)
        
    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        init.constant_(self.self_loop, 0)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, adj, feat):
        src_degrees = adj.sum(dim=0).clamp(min=1)
        dst_degrees = adj.sum(dim=1).clamp(min=1)
        loop = th.matmul(feat, self.weight)
        feat_src = feat

        if self._norm == "both":
            norm_src = th.pow(src_degrees, -0.5)
            shp = norm_src.shape + (1,) * (feat.dim() - 1)
            norm_src = th.reshape(norm_src, shp).to(feat.device)
            feat_src = feat_src * norm_src
        
        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat_src = th.matmul(feat_src, self.weight)
            rst = adj @ feat_src
        else:
            # aggregate first then mult W
            rst = adj @ feat_src
            rst = th.matmul(rst, self.weight)
            
        if self._norm != "none":
            if self._norm == "both":
                norm_dst = th.pow(dst_degrees, -0.5)
            else:  # right
                norm_dst = 1.0 / dst_degrees
            shp = norm_dst.shape + (1,) * (feat.dim() - 1)
            norm_dst = th.reshape(norm_dst, shp).to(feat.device)
            rst = rst * norm_dst
        
        rst = rst + self.self_loop * loop
        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)
        
        if self._in_feats == self._out_feats:
            rst = rst + feat
        return rst
    
class GCNSequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
    
    def forward(self, adj, X):
        for module in self:
            X = module(adj, X)
        return X

def BN_SELU(num_features):
    return nn.Sequential(nn.BatchNorm1d(num_features),
                         nn.SELU())