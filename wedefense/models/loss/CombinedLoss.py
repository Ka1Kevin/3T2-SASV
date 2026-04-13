import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CombinedLoss(nn.Module):
    def __init__(self, num_out, num_class, margin=0.2, scale=32, contrastive_margin=1.0):
        super().__init__()
        # ==== AAM-Softmax ====
        self.m = margin
        self.s = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_class, num_out))
        nn.init.xavier_normal_(self.weight)
        self.ce = nn.CrossEntropyLoss()

        # ==== Contrastive ====
        self.contrastive_margin = contrastive_margin

    def aam_softmax(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine**2).clamp(0,1))
        phi = cosine * math.cos(self.m) - sine * math.sin(self.m)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1,1), 1)
        logits = (one_hot * phi) + (1 - one_hot) * cosine
        logits *= self.s
        return self.ce(logits, label)

    def contrastive(self, x, label):
        # 简单形式: 假设输入 batch 是成对的 (2N)
        # label=1 表示同类，0 表示不同类
        z1, z2 = x[0::2], x[1::2]
        dist = F.pairwise_distance(z1, z2)
        return torch.mean(label.float() * dist.pow(2) + (1-label.float()) * F.relu(self.contrastive_margin - dist).pow(2))

    def forward(self, x, label):
        return self.aam_softmax(x, label) + self.contrastive(x, label)
