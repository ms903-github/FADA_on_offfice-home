import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import heapq
import itertools
from sklearn.cluster import KMeans


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def dist_distance(net_g, s_dataloader, t_dataloader, n_class, dist="l1_ave"):
    s_dist_sum = torch.cuda.FloatTensor(np.zeros((n_class, 2000)))
    t_dist_sum = torch.cuda.FloatTensor(np.zeros((n_class, 2000)))
    
    if dist == "l1_ave":
        #ソースのクラスごとの特徴量分布(平均)を計算
        j = 0
        for data, label in s_dataloader:
            data, label = data.to(device), label.to(device)
            s_dist = torch.cuda.FloatTensor(np.zeros((n_class, 2000)))
            _, _, feat = net_g(data)
            for i in range(len(data)):
                s_dist[int(label[i])] += feat[i]
            s_dist /= len(data)
            s_dist_sum += s_dist
        
        #ターゲットのクラスごとの特徴量分布(平均)を計算
        j = 0
        for data, label in t_dataloader:
            data, label = data.to(device), label.to(device)
            t_dist = torch.cuda.FloatTensor(np.zeros((n_class, 2000)))
            _, _, feat = net_g(data)
            for i in range(len(data)):
                t_dist[int(label[i])] += feat[i]
            t_dist /= len(data)
            t_dist_sum += t_dist

        loss = torch.mean(torch.abs(s_dist_sum - t_dist_sum))
        return loss
    
    #ソース・ターゲット分布同士のkldivergence
    if dist == "kl_div_st":
        loss = 0
        #source
        for data, label in s_dataloader:
            data, label = data.to(device), label.to(device)
            s_list = [[] for i in range(n_class)] 
            _, _, feat = net_g(data)
            for i in range(len(data)):
                s_list[int(label[i])].append(feat[i])
            
        #target
        for data, label in t_dataloader:
            data, label = data.to(device), label.to(device)
            t_list = [[] for i in range(n_class)]
            _, _, feat = net_g(data)
            for i in range(len(data)):
                t_list[int(label[i])].append(feat[i])
            
        for c in range(n_class):
            loss += F.kl_div(torch.FloatTensor(s_list[c]), torch.FloatTensor(t_list[c]))
            
        return loss

    #ソースにおける異なるクラス同士のkldivergence(これは大きい方が望ましい)
    if dist == "kl_div_s":
        #source
        for data, label in s_dataloader:
            data, label = data.to(device), label.to(device)
            s_list = [[] for i in range(n_class)] 
            _, _, feat = net_g(data)
            for i in range(len(data)):
                s_list[int(label[i])].append(feat[i])
        losses = []
        for c1, c2 in itertools.combinations(range(n_class), 2):
            losses.append(F.kl_div(torch.FloatTensor(s_list[c1]), torch.FloatTensor(s_list[c2])))
        loss = -sum(heapq.nsmallest(10, losses))
        return loss
        
    
def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, mutual_flag=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss