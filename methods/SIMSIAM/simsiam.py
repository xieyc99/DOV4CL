import os 
import sys 
import time 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

from warmup_scheduler import GradualWarmupScheduler

from methods.base import CLModel, CLTrainer
from utils.util import AverageMeter, save_model, load_model
from utils.knn import knn_monitor 

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

class SimSiamModel(CLModel):
    def __init__(self, args):
        super().__init__(args)

        if self.mlp_layers == 2:
            self.projector = nn.Sequential(
                    nn.Linear(self.feat_dim, self.feat_dim),
                    nn.BatchNorm1d(self.feat_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.feat_dim, self.feat_dim),
                    nn.BatchNorm1d(self.feat_dim),
                )
        elif self.mlp_layers == 3:
            self.projector = nn.Sequential(
                    nn.Linear(self.feat_dim, self.feat_dim),
                    nn.BatchNorm1d(self.feat_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.feat_dim, self.feat_dim),
                    nn.BatchNorm1d(self.feat_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.feat_dim, self.feat_dim),
                    nn.BatchNorm1d(self.feat_dim),
                )
            
        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        
        self.predictor = nn.Sequential(
                    nn.Linear(self.feat_dim, int(self.feat_dim/4)),
                    nn.BatchNorm1d(int(self.feat_dim/4)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(self.feat_dim/4), self.feat_dim),
                )




    @torch.no_grad()
    def moving_average(self):
        """
        Momentum update of the key encoder
        """
        m = 0.5
        for param_q, param_k in zip(self.distill_backbone.parameters(), self.backbone.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)
        
    def forward(self, v1, v2):
        f, h = self.encoder, self.predictor
        z1, z2 = f(v1), f(v2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return L
