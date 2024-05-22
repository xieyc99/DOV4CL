# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import random
from PIL import ImageFilter
import kornia
from methods.base import CLModel, CLTrainer

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, kernel_size=(3, 3), sigma=[.1, 2.]):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, x):
        blurred_image = kornia.filters.gaussian_blur2d(x, self.kernel_size, self.sigma)
        return blurred_image

class MoCov3(CLModel):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, args):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCov3, self).__init__(args)

        self.K = 65536
        self.m = 0.999
        self.T = 0.07

        self.criterion = nn.CrossEntropyLoss().cuda(args.gpu)

        # create the encoders
        # num_classes is the output fc dimension
        self.dim = 128
        self.encoder_q = self.model_fun(full_model=True, num_classes=self.dim, mocov3=True)
        self.encoder_k = self.model_fun(full_model=True, num_classes=self.dim, mocov3=True)


        # if True:  # hack: brute-force replacement
        #     dim_mlp = self.encoder_q.fc.weight.shape[1]
        #     self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        #     self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        self.predictor = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.BatchNorm1d(self.dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim, self.dim),
        )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x1, x2):
        """
        Input:
            x1: a batch of query images
            x2: a batch of key images
        Output:
            q1,q2,k1,k2
        """

        # compute query features
        q1, q2 = self.predictor(self.encoder_q(x1)), self.predictor(self.encoder_q(x2))

        q1 = nn.functional.normalize(q1, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k1, k2 = self.encoder_k(x1), self.encoder_k(x2)  # keys: NxC
            k1 = nn.functional.normalize(k1, dim=1)
            k2 = nn.functional.normalize(k2, dim=1)

        return q1, q2, k1, k2

    def ctr(self, q, k):
        logits = torch.mm(q, k.t())
        N = q.size(0)
        labels = range(N)
        labels = torch.LongTensor(labels).cuda()
        loss = self.criterion(logits/self.T, labels)
        return 2*self.T*loss


