# -*- coding: utf-8 -*-            
# Time : 2022/12/20 21:26
# Author : Regulus
# FileName: others_test.py
# Explain: 
# Software: PyCharm

import argparse
import time
import CKA
import numpy
import timm.models
import random as rd
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

from braincog.base.node.node import *
from braincog.utils import *
from braincog.base.utils.criterions import *
from braincog.datasets.datasets import *
from braincog.model_zoo.resnet import *
from braincog.model_zoo.convnet import *
from braincog.model_zoo.vgg_snn import VGG_SNN
from braincog.model_zoo.resnet19_snn import resnet19
from braincog.utils import save_feature_map, setup_seed
from braincog.base.utils.visualization import plot_tsne_3d, plot_tsne, plot_confusion_matrix

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from rgb_hsv import RGB_HSV
import matplotlib.pyplot as plt
from timm.data import ImageDataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import load_checkpoint, create_model, resume_checkpoint, convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

# from ptflops import get_model_complexity_info
# from thop import profile, clever_format
# x = torch.randn(1, 48*48*2)
# y = torch.randn(1, 48*48*2)
# cka_value = torch.abs(CKA.linear_CKA(x, y))
# print("CKA:{}".format(cka_value))

import numpy as np
def linear_kernel(X, Y):
    return np.matmul(X, Y.T)


def rbf(X, Y, sigma=None):
    """
    Radial-Basis Function kernel for X and Y with bandwith chosen
    from median if not specified.
    """
    GX = np.dot(X, Y.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def HSIC(K, L):
    """
    Calculate Hilbert-Schmidt Independence Criterion on K and L.
    """
    n = K.shape[0]
    H = np.identity(n) - (1. / n) * np.ones((n, n))

    KH = np.matmul(K, H)
    LH = np.matmul(L, H)
    return 1. / (((n - 1) ** 2) * np.trace(np.matmul(KH, LH)) + 1e-7)


def CKA(X, Y, kernel=None):
    """
    Calculate Centered Kernel Alingment for X and Y. If no kernel
    is specified, the linear kernel will be used.
    """
    kernel = linear_kernel if kernel is None else kernel

    K = kernel(X, X)
    L = kernel(Y, Y)

    hsic = HSIC(K, L)
    varK = np.sqrt(HSIC(K, K))
    varL = np.sqrt(HSIC(L, L))
    return hsic / (varK * varL)


n = 20   # Samples
p1 = 10 * 2 * 48 * 48   # Representation dim model 1
p2 = 10 * 2 * 48 * 48   # Representation dim model 1

# Generate X
X = np.random.normal(size=(n, p1))
Y = np.random.normal(size=(n, p2))

# Center columns
X = X - np.mean(X, 0)
Y = Y - np.mean(Y, 0)

print(f'Linear CKA, between X and Y: {CKA(X, Y):1.5f}')
print(f'Linear CKA, between X and X: {CKA(X, X):1.5f}')

print(f'RBF Kernel CKA, between X and Y: {CKA(X, Y, rbf):1.5f}')
print(f'RBF Kernel CKA, between X and X: {CKA(X, X, rbf):1.5f}')