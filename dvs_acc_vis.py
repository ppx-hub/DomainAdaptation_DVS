# -*- coding: utf-8 -*-            
# Time : 2022/11/12 17:40
# Author : Regulus
# FileName: dvs_acc_vis.py
# Explain:
# Software: PyCharm

import os
import sys

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
from matplotlib.pyplot import MultipleLocator
from pylab import *
# sns.set_style('darkgrid')
# sns.set_palette('muted')
# sns.set_style("whitegrid")
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
    dataset = 'omniglot'  # [dvsc10, NCALTECH101, omniglot]
    traindataratio = '1.0'
    # ax1 = ax.inset_axes([0.2, 0.3, 0.28, 0.22])
    # ax2 = ax.inset_axes([0.65, 0.3, 0.28, 0.22])
    ANN_VIS = False

    # ------ plot additional ---------
    epoch_list, acc_list = extract_csv("/home/hexiang/DomainAdaptation_DVS/Results/Baseline/VGG_SNN-NCALTECH101-10-seed"
                                       "_42-bs_120-DA_False-ls_0.0-lr_0.005-traindataratio_1.0-TET_loss_True-refined_True/summary.csv", type='main')
    ax.plot(range(200, 200 + len(epoch_list)), acc_list, linewidth=2, label="Finetune:81.23")
    plt.xlabel('Training epoch', fontsize=18)
    plt.ylabel('Test Accuracy (Test set)', fontsize=18)
    plt.grid(which='major', axis='x', ls=':', lw=0.8, color='c', alpha=0.5)
    plt.grid(which='major', axis='y', ls=':', lw=0.8, color='c', alpha=0.5)
    # xticks([x for x in range(0, len(epoch_list))], ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"))
    plt.xticks(np.linspace(200, 400, 6))
    # ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.axhline(79.89, marker="*", color="r", linestyle="dashdot", label="Baseline:79.89")
    ax.legend(bbox_to_anchor=(1, 0), loc=4, fontsize=13)
    # plt.text(350, 75, "Baseline:79.88", size=15, color="r",  weight="light", bbox=dict(facecolor="r", alpha=0.2))
    plt.savefig('acc_finetune.pdf', dpi=300)
    sys.exit()
    # ------ plot block a ------------
    # if dataset != 'omniglot':
    #     # seed_list = [42, 47, 1024, 114514]
    #     seed_list = [42, 1024, 114514]
    #     legend_list = ['baseline', "w/. domain loss", "w/. domain loss + semantic loss"]
    #     baseline_root = '/home/hexiang/TransferLearning_For_DVS/Results_lastest/Baseline/'
    #     trainresults_root = '/home/hexiang/TransferLearning_For_DVS/Results_lastest/train_TCKA_test/'
    #
    #     show_epoch = 0
    #     for i in range(1):
    #         epoch_lists = []
    #         acc_lists = []
    #         if i == 12:
    #             for seed in seed_list:
    #                 if dataset == 'dvsc10':
    #                     file = os.path.join(baseline_root, 'VGG_SNN-dvsc10-10-seed_{}-bs_120-DA_True-ls_0.0-lr_0.005-traindataratio_{}-TET_first_True-TET_second_True/summary.csv'.format(seed, traindataratio))
    #                 else:
    #                     file = os.path.join(baseline_root, 'VGG_SNN-NCALTECH101-10-seed_{}-bs_120-DA_False-ls_0.0-lr_0.005-traindataratio_{}-TET_first_True-TET_second_True/summary.csv'.format(seed, traindataratio))
    #                 epoch_list, acc_list = extract_csv(file, type='main')
    #                 epoch_lists.append(epoch_list)
    #                 acc_lists.append(acc_list)
    #         elif i == 9:
    #             for seed in seed_list:
    #                 if dataset == 'dvsc10':
    #                     file = os.path.join(trainresults_root,
    #                                         'Transfer_VGG_SNN-dvsc10-10-bs_120-seed_{}-DA_True-ls_0.0-lr_0.005-SNR_0-domainLoss_True-semanticLoss_False-domain_loss_coefficient1.0-semantic_loss_coefficient0.5-traindataratio_{}-TETfirst_True-TETsecond_True/summary.csv'.format(
    #                                             seed, traindataratio))
    #                 else:
    #                     file = os.path.join(trainresults_root,
    #                                         'Transfer_VGG_SNN-NCALTECH101-10-bs_120-seed_{}-DA_False-ls_0.0-lr_0.005-SNR_0-domainLoss_True-semanticLoss_False-domain_loss_coefficient1.0-semantic_loss_coefficient0.001-traindataratio_{}-TETfirst_True-TETsecond_True/summary.csv'.format(
    #                                             seed, traindataratio))
    #                 epoch_list, acc_list = extract_csv(file, type='transfer')
    #                 epoch_lists.append(epoch_list)
    #                 acc_lists.append(acc_list)
    #
    #         else:
    #             for seed in seed_list:
    #                 if dataset == 'dvsc10':
    #                     file = os.path.join(trainresults_root,
    #                                         'Transfer_VGG_SNN-dvsc10-10-bs_120-seed_{}-DA_True-ls_0.1-lr_0.005-SNR_0-domainLoss_True-semanticLoss_True-domain_loss_coefficient1.0-semantic_loss_coefficient0.5-traindataratio_{}-TETfirst_True-TETsecond_True/summary.csv'.format(
    #                                             seed, traindataratio))
    #                 else:
    #                     file = os.path.join(trainresults_root,
    #                                         'Transfer_VGG_SNN-NCALTECH101-10-bs_120-seed_{}-DA_False-ls_0.1-lr_0.005-SNR_0-domainLoss_True-semanticLoss_True-domain_loss_coefficient1.0-semantic_loss_coefficient0.001-traindataratio_{}-TETfirst_True-TETsecond_True/summary.csv'.format(
    #                                             seed, traindataratio))
    #                 epoch_list, acc_list = extract_csv(file, type='transfer')
    #                 epoch_lists.append(epoch_list)
    #                 acc_lists.append(acc_list)
    #
    #         acc_mean = np.max(np.array(acc_lists), axis=1)
    #         acc_std = np.max(np.array(acc_lists), axis=1).std(0)
    #         print("for {}, acc max:{}, acc max mean:{} acc var:{}".format(legend_list[i], acc_mean, acc_mean.mean(), acc_std))
    #         # print("acc list:{}".format(np.max(np.array(acc_lists), axis=1)))
    #
    #         acc_lists_mean_total = np.array(acc_lists).mean(0)[show_epoch:]
    #         acc_lists_std_total = np.array(acc_lists).std(0)[show_epoch:]
    #
    #         ax.plot(range(1, len(acc_lists_mean_total) + 1), acc_lists_mean_total, linewidth=2, label=legend_list[i])
    #         ax.fill_between(range(1, len(acc_lists_mean_total) + 1), (acc_lists_mean_total - 1 * acc_lists_std_total), (acc_lists_mean_total + 1 * acc_lists_std_total), alpha=.3)
    #         # plot small figure
    #         ax1.plot(range(1, len(acc_lists_mean_total) + 1), acc_lists_mean_total, linewidth=2, label=legend_list[i])
    #         ax1.fill_between(range(1, len(acc_lists_mean_total) + 1), (acc_lists_mean_total - 1 * acc_lists_std_total), (acc_lists_mean_total + 1 * acc_lists_std_total), alpha=.3)
    #         ax2.plot(range(1, len(acc_lists_mean_total) + 1), acc_lists_mean_total, linewidth=2, label=legend_list[i])
    #         ax2.fill_between(range(1, len(acc_lists_mean_total) + 1), (acc_lists_mean_total - 1 * acc_lists_std_total), (acc_lists_mean_total + 1 * acc_lists_std_total), alpha=.3)
    # else:
    #     seed_list = [42, 47, 114514]
    #     legend_list = ['baseline', "w/. domain loss + semantic loss"]
    #     baseline_root = '/home/hexiang/TransferLearning_For_DVS/Results_lastest/Baseline/'
    #     trainresults_root = '/home/hexiang/TransferLearning_For_DVS/Results_lastest/train_TCKA_test/'
    #
    #     show_epoch = 50
    #     for i in range(2):
    #         epoch_lists = []
    #         acc_lists = []
    #         if i == 0:
    #             for seed in seed_list:
    #                 file = os.path.join(baseline_root,
    #                                     'SCNN-nomni-12-seed_{}-bs_64-DA_False-ls_0.0-lr_0.01-traindataratio_1.0-TET_first_False-TET_second_False/summary.csv'.format(
    #                                         seed))
    #                 epoch_list, acc_list = extract_csv(file, type='main')
    #                 epoch_lists.append(epoch_list)
    #                 acc_lists.append(acc_list)
    #         else:
    #             for seed in seed_list:
    #                 file = os.path.join(trainresults_root,
    #                                     'Transfer_SCNN-nomni-12-bs_64-seed_{}-DA_False-ls_0.0-lr_0.01-m_-1.0-domainLoss_True-semanticLoss_True-domain_loss_coefficient1.0-semantic_loss_coefficient0.5-traindataratio_1.0-TETfirst_False-TETsecond_False/summary.csv'.format(
    #                                         seed))
    #                 epoch_list, acc_list = extract_csv(file, type='transfer')
    #                 epoch_lists.append(epoch_list)
    #                 acc_lists.append(acc_list)
    #
    #         acc_mean = np.max(np.array(acc_lists), axis=1)
    #         acc_std = np.max(np.array(acc_lists), axis=1).std(0)
    #         print("for {}, acc max:{}, acc max mean:{} acc var:{}".format(legend_list[i], acc_mean, acc_mean.mean(),
    #                                                                       acc_std))
    #         # print("acc list:{}".format(np.max(np.array(acc_lists), axis=1)))
    #
    #         # epoch_lists = np.array(epoch_lists).mean(0)[show_epoch:]
    #         # acc_lists_mean = np.array(acc_lists).mean(0)[show_epoch:]
    #         # acc_lists_std = np.array(acc_lists).std(0)[show_epoch:]
    #         #
    #         # ax.plot(range(1, len(acc_lists_mean) + 1), acc_lists_mean, linewidth=2, label=legend_list[i])
    #         # ax.fill_between(range(1, len(acc_lists_mean) + 1), (acc_lists_mean - 1 * acc_lists_std),
    #         #                 (acc_lists_mean + 1 * acc_lists_std), alpha=.3)
    #     sys.exit()
    #
    # ax1.set_xlim(550, 600)
    # if dataset == 'dvsc10':
    #     if traindataratio == '0.1':  # 0.1的时候, seed 47最好
    #         ax1.set_xlim(290, 310)
    #         ax2.set_xlim(550, 600)
    #         ax1.set_ylim(55, 65)
    #         ax2.set_ylim(58, 65)
    #     else:
    #         ax1.set_xlim(290, 310)
    #         ax2.set_xlim(580, 600)
    #         ax1.set_ylim(79, 85)
    #         ax2.set_ylim(80, 85)
    # else:
    #     if traindataratio == '0.1':
    #         ax1.set_ylim(40, 55)
    #     else:
    #         ax1.set_ylim(78.5, 82.9)
    # ax.indicate_inset_zoom(ax1)
    # ax.indicate_inset_zoom(ax2)
    # ax.legend(bbox_to_anchor=(1, 0), loc=4, borderaxespad=0)
    # plt.xlabel('Epochs', fontsize=15)
    # plt.ylabel('Test Accuracy (Test set)', fontsize=15)
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    # # plt.show()
    # plt.savefig('{}_{}.svg'.format(dataset, traindataratio), dpi=300)

# ------ plot block b ------------
way = "b"
if way == 'a':
    ncaltech101_acc_baseline = [45.40, 56.44, 64.83, 67.93, 71.72, 73.22, 76.78, 77.13, 78.74, 79.54]
    ncaltech101_acc_ours     = [55.17, 63.22, 68.39, 71.49, 74.25, 77.01, 79.20, 78.74, 79.20, 81.72]
    ax.plot(range(0, 10), ncaltech101_acc_baseline, '*', linewidth=2, label='Baseline', linestyle="dashdot")
    ax.plot(range(0, 10), ncaltech101_acc_ours, 'o', linewidth=2, label='Knowledge-Transfer (Ours)', linestyle="dashdot")
    ax.legend(bbox_to_anchor=(1, 0), loc=4, fontsize=13)
    plt.xlabel('Ratio of training data', fontsize=15)
    plt.ylabel('Test Accuracy (Test set)', fontsize=15)
    plt.grid(which='major', axis='x', ls=':', lw=0.8, color='c', alpha=0.5)
    plt.grid(which='major', axis='y', ls=':', lw=0.8, color='c', alpha=0.5)
    xticks([x for x in range(0, 10)], ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.savefig('ncaltech101_dataratio.pdf', dpi=300)
elif way == 'b':
    ncaltech101_acc_baseline = [45.40, 45.40, 45.40, 45.40, 45.40, 45.40]
    ncaltech101_acc_ours     = [45.40, 52.64, 54.36, 53.33, 55.17, 55.17]
    ax.plot(range(0, 6), ncaltech101_acc_baseline, '*', linewidth=2, label='Baseline', linestyle="dashdot")
    ax.plot(range(0, 6), ncaltech101_acc_ours, 'o', linewidth=2, label='Knowledge-Transfer (Ours)', linestyle="dashdot")
    ax.set_ylim([40, 60])
    ax.legend(bbox_to_anchor=(1, 0), loc=4, fontsize=13)
    plt.xlabel('Rgb data amout ratio', fontsize=15)
    plt.ylabel('Test Accuracy (Test set)', fontsize=15)
    plt.grid(which='major', axis='x', ls=':', lw=0.8, color='c', alpha=0.5)
    plt.grid(which='major', axis='y', ls=':', lw=0.8, color='c', alpha=0.5)
    xticks([x for x in range(0, 6)], ("0%", "5%", "10%", "50%", "80%", "100%"))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.savefig('ncaltech101_dvsdataratio.pdf', dpi=600)
elif way == 'c':
    x_data = ['Baseline', 'w/o v channel', 'w/ v channel']
    ax_data = [83.60, 84.20, 84.50]
    ax1_data = [79.54, 81.26, 81.72]

    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)

    ax.set_ylim([83, 85])
    ax.set_yticks = np.arange(83, 85)
    ax.set_yticklabels = np.arange(83, 85)

    bar_width = 0.25
    ax.set_ylabel('Test Accuracy (Test set)', fontsize=12);
    lns1 = ax.bar(x=np.arange(len(x_data)), width=bar_width, height=ax_data, label='CIFAR10-DVS', fc='steelblue', alpha=0.8)

    for a, b in enumerate(ax_data):
        plt.text(a, b + 0.0005, '%s' % b, ha='center', fontsize=9)

    ax1 = ax.twinx()  # this is the important function

    ax1.set_ylim([79, 82])
    ax1.set_yticks = np.arange(79, 82)
    ax1.set_yticklabels = np.arange(79, 82)
    ax1.set_ylabel('Test Accuracy (Test set)', fontsize=12);
    lns2 = ax1.bar(x=np.arange(len(x_data)) + bar_width, width=bar_width, height=ax1_data, label='N-Caltech101', fc='indianred',
                   alpha=0.8)

    for a, b in enumerate(ax1_data):
        plt.text(a + 0.25, b + 0.001, '%s' % b, ha='center', fontsize=9)

    plt.xticks(np.arange(len(x_data)) + bar_width / 2, x_data)
    plt.grid(which='major', axis='y', ls=':', lw=0.8, color='c', alpha=0.35)
    ax.set_xlabel('Methods', fontsize=12)

    fig.legend(loc=1, bbox_to_anchor=(0.35, 1), bbox_transform=ax.transAxes, fontsize=9)
    plt.tight_layout()  # 超出边界加上的
    plt.savefig('v_channel.pdf', dpi=600)  # 图表输出
