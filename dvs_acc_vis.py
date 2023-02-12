# -*- coding: utf-8 -*-            
# Time : 2022/11/12 17:40
# Author : Regulus
# FileName: dvs_acc_vis.py
# Explain:
# Software: PyCharm

import os
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from tonic.datasets import NCALTECH101, CIFAR10DVS
from braincog.datasets.datasets import unpack_mix_param, DATA_DIR
from braincog.datasets.cut_mix import *
import tonic
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter

import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1,
                rc={"lines.linewidth": 2.5})

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
color_lib = sns.color_palette()

def extract_csv(file, type='main'):
    if type == 'main':
        column = [0, 3]
    elif type == 'transfer':
        column = [0, 7]
    data = np.loadtxt(open(file, "rb"), delimiter=",", skiprows=1, usecols=column)
    return data[:, 0], data[:, 1]


def to_percent(temp, position):
    return '%2.f'%(temp) + '%'


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 16
    fig, ax = plt.subplots(figsize=(10, 6))

    dataset = 'dvsc10'  # [dvsc10, NCALTECH101]
    ANN_VIS = False
    seed_list = [42, 47, 1024]
    legend_list = ['baseline', "w/. domain loss + semantic loss", "w/. domain loss + semantic loss + ls"]
    baseline_root = '/home/hexiang/TransferLearning_For_DVS/Results_new_refined/Baseline/'
    trainresults_root = '/home/hexiang/TransferLearning_For_DVS/Results_new_refined/train_TCKA_test/'
    show_epoch = 450

    for i in range(3):
        epoch_lists = []
        acc_lists = []
        if i == 0:
            for seed in seed_list:
                if dataset == 'dvsc10':
                    file = os.path.join(baseline_root, 'VGG_SNN-dvsc10-10-seed_{}-bs_120-DA_True-ls_0.0-traindataratio_1.0/summary.csv'.format(seed))
                else:
                    file = os.path.join(baseline_root, 'VGG_SNN-NCALTECH101-10-seed_{}-bs_120-DA_False-ls_0.0-traindataratio_1.0/summary.csv'.format(seed))
                epoch_list, acc_list = extract_csv(file, type='main')
                epoch_lists.append(epoch_list)
                acc_lists.append(acc_list)
        elif i== 1:
            for seed in seed_list:
                if dataset == 'dvsc10':
                    file = os.path.join(trainresults_root,
                                        'Transfer_VGG_SNN-dvsc10-10-bs_120-seed_{}-DA_True-ls_0.0-SNR_0-domainLoss_True-semanticLoss_True-domain_loss_coefficient1.0-semantic_loss_coefficient1.0-traindataratio_1.0-lossafter_False/summary.csv'.format(
                                            seed))
                else:
                    file = os.path.join(trainresults_root,
                                        'Transfer_VGG_SNN-NCALTECH101-10-bs_120-seed_{}-DA_False-ls_0.0-SNR_0-domainLoss_True-semanticLoss_True-domain_loss_coefficient1.0-semantic_loss_coefficient0.001-traindataratio_1.0-lossafter_False/summary.csv'.format(
                                            seed))
                epoch_list, acc_list = extract_csv(file, type='transfer')
                epoch_lists.append(epoch_list)
                acc_lists.append(acc_list)

        else:
            for seed in seed_list:
                if dataset == 'dvsc10':
                    file = os.path.join(trainresults_root,
                                        'Transfer_VGG_SNN-dvsc10-10-bs_120-seed_{}-DA_True-ls_0.1-SNR_0-domainLoss_True-semanticLoss_True-domain_loss_coefficient1.0-semantic_loss_coefficient1.0-traindataratio_1.0-lossafter_False/summary.csv'.format(
                                            seed))
                else:
                    file = os.path.join(trainresults_root,
                                        'Transfer_VGG_SNN-NCALTECH101-10-bs_120-seed_{}-DA_False-ls_0.1-SNR_0-domainLoss_True-semanticLoss_True-domain_loss_coefficient1.0-semantic_loss_coefficient0.001-traindataratio_1.0-lossafter_False/summary.csv'.format(
                                            seed))
                epoch_list, acc_list = extract_csv(file, type='transfer')
                epoch_lists.append(epoch_list)
                acc_lists.append(acc_list)

        acc_mean = np.max(np.array(acc_lists), axis=1)
        acc_std = np.max(np.array(acc_lists), axis=1).std(0)
        print("for {}, acc mean:{}, acc var:{}".format(legend_list[i], acc_mean, acc_std))
        # print("acc list:{}".format(np.max(np.array(acc_lists), axis=1)))

        epoch_lists = np.array(epoch_lists).mean(0)[show_epoch:]
        acc_lists_mean = np.array(acc_lists).mean(0)[show_epoch:]
        acc_lists_std = np.array(acc_lists).std(0)[show_epoch:]

        ax.plot(range(1, len(acc_lists_mean) + 1), acc_lists_mean, linewidth=2, label=legend_list[i])
        ax.fill_between(range(1, len(acc_lists_mean) + 1), (acc_lists_mean - 1 * acc_lists_std), (acc_lists_mean + 1 * acc_lists_std), alpha=.3)

    ax.legend(bbox_to_anchor=(1, 0), loc=4, borderaxespad=0)
    plt.xlabel('Training epochs in {}'.format(dataset), fontsize=20)
    plt.ylabel('Accuracy (Test set)', fontsize=20)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.show()