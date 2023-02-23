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
    dataset = 'dvsc10'  # [dvsc10, NCALTECH101, omniglot]
    traindataratio = '0.1'
    ax1 = ax.inset_axes([0.2, 0.3, 0.28, 0.22])
    ax2 = ax.inset_axes([0.65, 0.3, 0.28, 0.22])
    ANN_VIS = False

    if dataset != 'omniglot':
        # seed_list = [42, 47, 114514]
        seed_list = [47]
        legend_list = ['baseline', "w/. domain loss", "w/. domain loss + semantic loss"]
        baseline_root = '/home/hexiang/TransferLearning_For_DVS/Results_lastest/Baseline/'
        trainresults_root = '/home/hexiang/TransferLearning_For_DVS/Results_lastest/train_TCKA_test/'

        show_epoch = 0
        for i in range(3):
            epoch_lists = []
            acc_lists = []
            if i == 0:
                for seed in seed_list:
                    if dataset == 'dvsc10':
                        file = os.path.join(baseline_root, 'VGG_SNN-dvsc10-10-seed_{}-bs_120-DA_True-ls_0.0-lr_0.005-traindataratio_{}-TET_first_True-TET_second_True/summary.csv'.format(seed, traindataratio))
                    else:
                        file = os.path.join(baseline_root, 'VGG_SNN-NCALTECH101-10-seed_{}-bs_120-DA_False-ls_0.0-lr_0.005-traindataratio_{}-TET_first_True-TET_second_True/summary.csv'.format(seed, traindataratio))
                    epoch_list, acc_list = extract_csv(file, type='main')
                    epoch_lists.append(epoch_list)
                    acc_lists.append(acc_list)
            elif i == 1:
                for seed in seed_list:
                    if dataset == 'dvsc10':
                        file = os.path.join(trainresults_root,
                                            'Transfer_VGG_SNN-dvsc10-10-bs_120-seed_{}-DA_True-ls_0.0-lr_0.005-SNR_0-domainLoss_True-semanticLoss_False-domain_loss_coefficient1.0-semantic_loss_coefficient0.5-traindataratio_{}-TETfirst_True-TETsecond_True/summary.csv'.format(
                                                seed, traindataratio))
                    else:
                        file = os.path.join(trainresults_root,
                                            'Transfer_VGG_SNN-NCALTECH101-10-bs_120-seed_{}-DA_False-ls_0.0-lr_0.005-SNR_0-domainLoss_True-semanticLoss_False-domain_loss_coefficient1.0-semantic_loss_coefficient0.001-traindataratio_{}-TETfirst_True-TETsecond_True/summary.csv'.format(
                                                seed, traindataratio))
                    epoch_list, acc_list = extract_csv(file, type='transfer')
                    epoch_lists.append(epoch_list)
                    acc_lists.append(acc_list)

            else:
                for seed in seed_list:
                    if dataset == 'dvsc10':
                        file = os.path.join(trainresults_root,
                                            'Transfer_VGG_SNN-dvsc10-10-bs_120-seed_{}-DA_True-ls_0.0-lr_0.005-SNR_0-domainLoss_True-semanticLoss_True-domain_loss_coefficient1.0-semantic_loss_coefficient0.5-traindataratio_{}-TETfirst_True-TETsecond_True/summary.csv'.format(
                                                seed, traindataratio))
                    else:
                        file = os.path.join(trainresults_root,
                                            'Transfer_VGG_SNN-NCALTECH101-10-bs_120-seed_{}-DA_False-ls_0.0-lr_0.005-SNR_0-domainLoss_True-semanticLoss_True-domain_loss_coefficient1.0-semantic_loss_coefficient0.001-traindataratio_{}-TETfirst_True-TETsecond_True/summary.csv'.format(
                                                seed, traindataratio))
                    epoch_list, acc_list = extract_csv(file, type='transfer')
                    epoch_lists.append(epoch_list)
                    acc_lists.append(acc_list)

            acc_mean = np.max(np.array(acc_lists), axis=1)
            acc_std = np.max(np.array(acc_lists), axis=1).std(0)
            print("for {}, acc max:{}, acc max mean:{} acc var:{}".format(legend_list[i], acc_mean, acc_mean.mean(), acc_std))
            # print("acc list:{}".format(np.max(np.array(acc_lists), axis=1)))

            acc_lists_mean_total = np.array(acc_lists).mean(0)[show_epoch:]
            acc_lists_std_total = np.array(acc_lists).std(0)[show_epoch:]

            ax.plot(range(1, len(acc_lists_mean_total) + 1), acc_lists_mean_total, linewidth=2, label=legend_list[i])
            ax.fill_between(range(1, len(acc_lists_mean_total) + 1), (acc_lists_mean_total - 1 * acc_lists_std_total), (acc_lists_mean_total + 1 * acc_lists_std_total), alpha=.3)
            # plot small figure
            ax1.plot(range(1, len(acc_lists_mean_total) + 1), acc_lists_mean_total, linewidth=2, label=legend_list[i])
            ax1.fill_between(range(1, len(acc_lists_mean_total) + 1), (acc_lists_mean_total - 1 * acc_lists_std_total), (acc_lists_mean_total + 1 * acc_lists_std_total), alpha=.3)
            ax2.plot(range(1, len(acc_lists_mean_total) + 1), acc_lists_mean_total, linewidth=2, label=legend_list[i])
            ax2.fill_between(range(1, len(acc_lists_mean_total) + 1), (acc_lists_mean_total - 1 * acc_lists_std_total), (acc_lists_mean_total + 1 * acc_lists_std_total), alpha=.3)
    else:
        seed_list = [42, 47, 52]
        legend_list = ['baseline', "w/. domain loss + semantic loss"]
        baseline_root = '/home/hexiang/TransferLearning_For_DVS/Results_new_refined/Baseline/'
        trainresults_root = '/home/hexiang/TransferLearning_For_DVS/Results_new_refined/train_TCKA_test/'

        show_epoch = 50
        for i in range(2):
            epoch_lists = []
            acc_lists = []
            if i == 0:
                for seed in seed_list:
                    file = os.path.join(baseline_root,
                                        'SCNN-nomni-12-seed_{}-bs_16-DA_False-ls_0.0-traindataratio_1.0/summary.csv'.format(
                                            seed))
                    epoch_list, acc_list = extract_csv(file, type='main')
                    epoch_lists.append(epoch_list)
                    acc_lists.append(acc_list)
            else:
                for seed in seed_list:
                    file = os.path.join(trainresults_root,
                                        'Transfer_SCNN-nomni-12-bs_16-seed_{}-DA_False-ls_0.0-SNR_0-domainLoss_True-semanticLoss_True-domain_loss_coefficient1.0-semantic_loss_coefficient1.0-traindataratio_1.0-lossafter_False/summary.csv'.format(
                                            seed))
                    epoch_list, acc_list = extract_csv(file, type='transfer')
                    epoch_lists.append(epoch_list)
                    acc_lists.append(acc_list)

            acc_mean = np.max(np.array(acc_lists), axis=1)
            acc_std = np.max(np.array(acc_lists), axis=1).std(0)
            print("for {}, acc max:{}, acc max mean:{} acc var:{}".format(legend_list[i], acc_mean, acc_mean.mean(),
                                                                          acc_std))
            # print("acc list:{}".format(np.max(np.array(acc_lists), axis=1)))

            epoch_lists = np.array(epoch_lists).mean(0)[show_epoch:]
            acc_lists_mean = np.array(acc_lists).mean(0)[show_epoch:]
            acc_lists_std = np.array(acc_lists).std(0)[show_epoch:]

            ax.plot(range(1, len(acc_lists_mean) + 1), acc_lists_mean, linewidth=2, label=legend_list[i])
            ax.fill_between(range(1, len(acc_lists_mean) + 1), (acc_lists_mean - 1 * acc_lists_std),
                            (acc_lists_mean + 1 * acc_lists_std), alpha=.3)

    ax1.set_xlim(550, 600)
    if dataset == 'dvsc10':
        if traindataratio == '0.1':  # 0.1的时候, seed 47最好
            ax1.set_xlim(290, 310)
            ax2.set_xlim(550, 600)
            ax1.set_ylim(55, 65)
            ax2.set_ylim(58, 65)
        else:
            ax1.set_xlim(290, 310)
            ax2.set_xlim(580, 600)
            ax1.set_ylim(79, 85)
            ax2.set_ylim(80, 85)
    else:
        if traindataratio == '0.1':
            ax1.set_ylim(40, 55)
        else:
            ax1.set_ylim(78.5, 82.9)
    ax.indicate_inset_zoom(ax1)
    ax.indicate_inset_zoom(ax2)
    ax.legend(bbox_to_anchor=(1, 0), loc=4, borderaxespad=0)
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Test Accuracy (Test set)', fontsize=15)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.show()