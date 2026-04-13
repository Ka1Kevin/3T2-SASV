#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class LossFunction(nn.Module):

    def __init__(self, init_w1=10.0, init_b1=-5.0, init_w2=10.0, init_b2=-5.0, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True       
        self.w1 = nn.Parameter(torch.tensor(init_w1))
        self.b1 = nn.Parameter(torch.tensor(init_b1))
        self.criterion  = torch.nn.CrossEntropyLoss()
        print('Initialized AngleProto')

    def forward(self, x, label=None, num_bna=0): # x(Batch_size, 2, Embedding_dim)
        assert x.size()[1] >= 2 

        out_anchor = x[:, 1, :]
        out_positive = x[:, 0, :][ :num_bna]
        stepsize = out_positive.size()[0]

        cos_sim_matrix1 = F.cosine_similarity(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))
        torch.clamp(self.w1, 1e-6)
        cos_sim_matrix1 = cos_sim_matrix1 * self.w1 + self.b1       

        out_anchor = x[:, 0, :]
        out_positive = x[:, 1, :][ :num_bna]
        
        label = torch.from_numpy(numpy.asarray(range(0,stepsize))).cuda()
        nloss1 = self.criterion(cos_sim_matrix1, label)
        nloss = nloss1

        prec1 = accuracy(cos_sim_matrix1.detach(), label.detach(), topk=(1,))[0]
        prec = prec1

        return nloss, prec