from __future__ import print_function
from __future__ import division
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import Parameter
import math

googlenet = models.googlenet(pretrained = True)


class Encoder_resnet(nn.Module):
    def __init__(self):
        super(Encoder_resnet, self).__init__()
        self.resnet = models.resnet50(pretrained = True)
        self.fc1 = nn.Linear(1000, 2000)
        self.fc2 = nn.Linear(2000, 2000)
        self.batchnorm = nn.BatchNorm1d(2000)
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1(x)
        x = self.batchnorm(x)
        x = self.fc2(x)
        return x

class Encoder_google(nn.Module):
    def __init__(self):
        super(Encoder_google, self).__init__()
        self.head = nn.Sequential(
            googlenet.conv1,
            googlenet.maxpool1,
            googlenet.conv2,
            googlenet.conv3,
            googlenet.maxpool2
            )
        self.i3 = nn.Sequential(
            googlenet.inception3a,
            googlenet.inception3b,
            googlenet.maxpool3
            )
        self.i4 = nn.Sequential(
            googlenet.inception4a,
            googlenet.inception4b,
            googlenet.inception4c,
            googlenet.inception4d,
            googlenet.inception4e,
            googlenet.maxpool4
            )
        self.i5 = nn.Sequential(
            googlenet.inception5a,
            googlenet.inception5b,
            googlenet.avgpool,
            googlenet.dropout
            )
        self.fc0 = googlenet.fc
        self.fc1 = nn.Linear(1000, 2000)
        self.fc2 = nn.Linear(2000, 2000)
        self.batchnorm = nn.BatchNorm1d(2000)

    def forward(self, x):
        x_h = F.relu(self.head(x))
        x = F.relu(self.i3(x_h))
        x_m = F.relu(self.i4(x))
        x = F.relu(self.i5(x_m))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.batchnorm(x)
        x_l = F.relu(self.fc2(x))

        return x_h, x_m, x_l

class classifier(nn.Module):     #classifier:h
    def __init__(self, n_class):
        super(classifier, self).__init__()
        self.fc1 = nn.Linear(2000, 1000)
        self.fc2 = nn.Linear(1000, n_class)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class classifier_arc(nn.Module):
    def __init__(self, n_class):
        super(classifier_arc, self).__init__()
        self.fc1 = nn.Linear(2000, 1000)
        self.arc = ArcMarginProduct(1000, n_class)
    
    def forward(self, x, label):
        x = self.fc1(x)
        return self.arc(x, label)

class classifier_cos(nn.Module):
    def __init__(self, n_class):
        super(classifier_cos, self).__init__()
        self.fc1 = nn.Linear(2000, 1000)
        self.cos = AddMarginProduct(1000, n_class)
    
    def forward(self, x, label):
        x = self.fc1(x)
        return self.cos(x, label)

class classifier_adacos(nn.Module):
    def __init__(self, n_class):
        super(classifier_adacos, self).__init__()
        self.fc1 = nn.Linear(2000, 1000)
        self.adacos = AdaCos(1000, n_class)
    
    def forward(self, x, label=None):
        x = F.relu(self.fc1(x))
        return self.adacos(x, label=label)

class classifier_sphere(nn.Module):
    def __init__(self, n_class):
        super(classifier_sphere, self).__init__()
        self.fc1 = nn.Linear(2000, 1000)
        self.sphere = SphereProduct(1000, n_class)
    
    def forward(self, x, label):
        x = self.fc1(x)
        return self.sphere(x, label)

class classifier_adacos_middle(nn.Module):
    def __init__(self, n_class):
        super(classifier_adacos_middle, self).__init__()
        self.conv1 = nn.Conv2d(832, 256, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        self.fc1 = nn.Linear(1024, 1000)
        self.adacos = AdaCos(1000, n_class)
    
    def forward(self, x, label):
        # N x 1056 x 14 x 14
        x = self.conv1(x)
        # N x 256 x ? x ?
        x = self.maxpool1(x)
        # N x 256 x 16 x 16 (?)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        return self.adacos(x, label=label)


class DCD_head(nn.Module):
    def __init__(self):
        super(DCD_head, self).__init__()
        self.conv1 = nn.Conv2d(384, 64, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        self.fc1 = nn.Linear(9216, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        # N x 384 x 28 x 28
        x = self.conv1(x)
        # N x 64 x 26 x 26
        x = self.maxpool1(x)
        # N x 64 x 13 x 13 (?)
        x = x.view(-1, 9216)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return(x)


class DCD_middle(nn.Module):
    def __init__(self):
        super(DCD_middle, self).__init__()
        self.conv1 = nn.Conv2d(1664, 256, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        self.fc1 = nn.Linear(9126, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        # N x 1056 x 14 x 14
        x = self.conv1(x)
        # N x 256 x 12 x 12
        x = self.maxpool1(x)
        # N x 256 x 6 x 6 (?)
        x = x.view(-1, 9126)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return(x)


class DCD_last(nn.Module):
    def __init__(self):
        super(DCD_last, self).__init__()
        self.fc1 = nn.Linear(4000, 512)        #1024が２つで2048
        self.fc2 = nn.Linear(512, 65)         
        self.fc3 = nn.Linear(65, 4)
        #self.dropout = nn.Dropout()
        self.batchnorm = nn.BatchNorm1d(65)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = self.batchnorm(x)
        return self.fc3(x)

class DomainDiscriminator(nn.Module):
    def __init__(self):
        super(DomainDiscriminator, self).__init__()
        self.fc1 = nn.Linear(2000, 512)        #1024が２つで2048
        self.fc2 = nn.Linear(512, 65)         
        self.fc3 = nn.Linear(65, 2)
        #self.dropout = nn.Dropout()
        self.batchnorm = nn.BatchNorm1d(65)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = self.batchnorm(x)
        return self.fc3(x)

"""
cited from github:
https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py

"""

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class SphereProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """
    def __init__(self, in_features, out_features, m=4):
        super(SphereProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'

"""
cited from
https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py

"""

class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return output
