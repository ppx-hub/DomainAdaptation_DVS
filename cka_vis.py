import copy
import os
import sys

import numpy as np
import torch
from scipy import stats
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
from tqdm import tqdm
import argparse
import time
import CKA
import timm.models
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

from timm.data import ImageDataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import load_checkpoint, create_model, resume_checkpoint, convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

# from ptflops import get_model_complexity_info
# from thop import profile, clever_format

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='SNN Training and Evaluating')

# Model parameters
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--dataset-two', default='cifar10', type=str)
parser.add_argument('--model', default='cifar_convnet', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--eval_checkpoint', default='', type=str, metavar='PATH',
                    help='path to eval checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')

# Dataset parameters for static datasets
parser.add_argument('--img-size', type=int, default=224, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='inputs image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')

# Dataloader parameters
parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='inputs batch size for training (default: 128)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.01,
                    help='weight decay (default: 0.01 for adamw)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--adam-epoch', type=int, default=1000, help='lamb switch to adamw')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=5e-3, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=600, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')
parser.add_argument('--power', type=int, default=1, help='power')

# Augmentation & regularization parameters ONLY FOR IMAGE NET
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.0)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--newton-maxiter', default=20, type=int,
                    help='max iterration in newton method')
parser.add_argument('--reset-drop', action='store_true', default=False,
                    help='whether to reset drop')
parser.add_argument('--kernel-method', type=str, default='cuda', choices=['torch', 'cuda'],
                    help='The implementation way of gaussian kernel method, choose from "cuda" and "torch"')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between node after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.99996,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of inputs bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='/home/hexiang/DomainAdaptation_DVS/Results/', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--device', type=int, default=0)

# Spike parameters
parser.add_argument('--step', type=int, default=10, help='Simulation time step (default: 10)')
parser.add_argument('--encode', type=str, default='direct', help='Input encode method (default: direct)')
parser.add_argument('--temporal-flatten', action='store_true',
                    help='Temporal flatten to channels. ONLY FOR EVENT DATA TRAINING BY ANN')
parser.add_argument('--adaptive-node', action='store_true')
parser.add_argument('--critical-loss', action='store_true')

# neuron type
parser.add_argument('--node-type', type=str, default='LIFNode', help='Node type in network (default: PLIF)')
parser.add_argument('--act-fun', type=str, default='GateGrad',
                    help='Surogate Function in node. Only for Surrogate nodes (default: AtanGrad)')
parser.add_argument('--threshold', type=float, default=.5, help='Firing threshold (default: 0.5)')
parser.add_argument('--tau', type=float, default=2., help='Attenuation coefficient (default: 2.)')
parser.add_argument('--requires-thres-grad', action='store_true')
parser.add_argument('--sigmoid-thres', action='store_true')

parser.add_argument('--loss-fn', type=str, default='ce', help='loss function (default: ce)')
parser.add_argument('--noisy-grad', type=float, default=0.,
                    help='Add noise to backward, sometime will make higher accuracy (default: 0.)')
parser.add_argument('--spike-output', action='store_true', default=False,
                    help='Using mem output or spike output (default: False)')
parser.add_argument('--n_groups', type=int, default=1)

# EventData Augmentation
parser.add_argument('--mix-up', action='store_true', help='Mix-up for event data (default: False)')
parser.add_argument('--cut-mix', action='store_true', help='CutMix for event data (default: False)')
parser.add_argument('--event-mix', action='store_true', help='EventMix for event data (default: False)')
parser.add_argument('--cutmix_beta', type=float, default=1.0, help='cutmix_beta (default: 1.)')
parser.add_argument('--cutmix_prob', type=float, default=0.5, help='cutmix_prib for event data (default: .5)')
parser.add_argument('--cutmix_num', type=int, default=1, help='cutmix_num for event data (default: 1)')
parser.add_argument('--cutmix_noise', type=float, default=0.,
                    help='Add Pepper noise after mix, sometimes work (default: 0.)')
parser.add_argument('--gaussian-n', type=int, default=3)
parser.add_argument('--rand-aug', action='store_true',
                    help='Rand Augment for Event data (default: False)')
parser.add_argument('--randaug_n', type=int, default=3,
                    help='Rand Augment times n (default: 3)')
parser.add_argument('--randaug_m', type=int, default=15,
                    help='Rand Augment times n (default: 15) (0-30)')
parser.add_argument('--train-portion', type=float, default=0.9,
                    help='Dataset portion, only for datasets which do not have validation set (default: 0.9)')
parser.add_argument('--event-size', default=48, type=int,
                    help='Event size. Resize event data before process (default: 48)')
parser.add_argument('--layer-by-layer', action='store_true',
                    help='forward step-by-step or layer-by-layer. '
                         'Larger Model with layer-by-layer will be faster (default: False)')
parser.add_argument('--node-resume', type=str, default='',
                    help='resume weights in node for adaptive node. (default: False)')
parser.add_argument('--node-trainable', action='store_true')

# visualize
parser.add_argument('--visualize', action='store_true',
                    help='Visualize spiking map for each layer, only for validate (default: False)')
parser.add_argument('--spike-rate', action='store_true',
                    help='Print spiking rate for each layer, only for validate(default: False)')
parser.add_argument('--tsne', action='store_true')
parser.add_argument('--conf-mat', action='store_true')

parser.add_argument('--suffix', type=str, default='',
                    help='Add an additional suffix to the save path (default: \'\')')

# for reconstructing es-imagenet
parser.add_argument('--reconstructed', action='store_true',
                    help='for ES-imagenet dataset')

parser.add_argument('--DVS-DA', action='store_true',
                    help='use DA on DVS')

# train data used ratio
parser.add_argument('--traindata-ratio', default=1.0, type=float,
                    help='training data ratio')

# use TET loss or not (all default False, do not use)
parser.add_argument('--TET-loss-first', action='store_true',
                    help='use TET loss one part')

parser.add_argument('--TET-loss-second', action='store_true',
                    help='use TET loss two part')

parser.add_argument('--no-use-hsv', action='store_true',
                    help='do not use hsv')

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

CALTECH101_list = []
CEPDVS_list = []

def main():
    args, args_text = _parse_args()
    # args.no_spike_output = args.no_spike_output | args.cut_mix
    args.no_spike_output = True
    output_dir = ''
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([
            args.model,
            args.dataset,
            str(args.step),
            "seed_{}".format(args.seed)
        ])
        output_dir = get_outdir(output_base, 'Baseline', exp_name)
        args.output_dir = output_dir
        setup_default_logging(log_path=os.path.join(output_dir, 'logx.txt'))

    else:
        setup_default_logging()

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.distributed and args.num_gpu > 1:
            _logger.warning(
                'Using more than one GPU per process in distributed mode is not allowed.Setting num_gpu to 1.')
            args.num_gpu = 1

    # args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.num_gpu = 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    else:
        torch.cuda.set_device('cuda:%d' % args.device)
    assert args.rank >= 0

    if args.distributed:
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on %d GPUs.' % args.num_gpu)

    # torch.manual_seed(args.seed + args.rank)
    setup_seed(args.seed + args.rank)

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        adaptive_node=args.adaptive_node,
        dataset=args.dataset,
        step=args.step,
        encode_type=args.encode,
        node_type=eval(args.node_type),
        threshold=args.threshold,
        tau=args.tau,
        sigmoid_thres=args.sigmoid_thres,
        requires_thres_grad=args.requires_thres_grad,
        spike_output=not args.no_spike_output,
        act_fun=args.act_fun,
        temporal_flatten=args.temporal_flatten,
        layer_by_layer=args.layer_by_layer,
        n_groups=args.n_groups,
        reconstruct=args.reconstructed,
        TET_loss=args.TET_loss_first or args.TET_loss_second
    )

    if 'dvs' in args.dataset:
        args.channels = 2
    elif 'mnist' in args.dataset:
        args.channels = 1
    else:
        args.channels = 3
    # flops, params = profile(model, inputs=(torch.randn(1, args.channels, args.event_size, args.event_size),), verbose=False)
    # _logger.info('flops = %fM', flops / 1e6)
    # _logger.info('param size = %fM', params / 1e6)

    linear_scaled_lr = args.lr * args.batch_size * args.world_size / 1024.0
    args.lr = linear_scaled_lr
    _logger.info("learning rate is %f" % linear_scaled_lr)

    if args.local_rank == 0:
        _logger.info('Model %s created, param count: %d' %
                     (args.model, sum([m.numel() for m in model.parameters()])))

    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    use_amp = None
    if args.amp:
        # for backwards compat, `--amp` arg tries apex before native amp
        if has_apex:
            args.apex_amp = True
        elif has_native_amp:
            args.native_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    if args.num_gpu > 1:
        if use_amp == 'apex':
            _logger.warning(
                'Apex AMP does not work well with nn.DataParallel, disabling. Use DDP or Torch AMP.')
            use_amp = None
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
        assert not args.channels_last, "Channels last not supported with DP, use DDP."
    else:
        model = model.cuda()
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)

    optimizer = create_optimizer(args, model)

    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume and args.eval_checkpoint == '':
        args.eval_checkpoint = args.resume
    if args.resume:
        args.eval = True
        # checkpoint = torch.load(args.resume, map_location='cpu')
        # model.load_state_dict(checkpoint['state_dict'], False)
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)
        # print(model.get_attr('mu'))
        # print(model.get_attr('sigma'))

    if args.critical_loss or args.spike_rate:
        model.set_requires_fp(True)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=args.resume)

    if args.node_resume:
        ckpt = torch.load(args.node_resume, map_location='cpu')
        model.load_node_weight(ckpt, args.node_trainable)

    model_without_ddp = model
    if args.distributed:
        if args.sync_bn:
            assert not args.split_bn
            try:
                if has_apex and use_amp != 'native':
                    # Apex SyncBN preferred unless native amp is activated
                    model = convert_syncbn_model(model)
                else:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if args.local_rank == 0:
                    _logger.info(
                        'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                        'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')
            except Exception as e:
                _logger.error('Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1')
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank],
                              find_unused_parameters=True)  # can use device str in Torch >= 1.1
        model_without_ddp = model.module
    # NOTE: EMA model does not need to be wrapped by DDP

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # now config only for imnet
    data_config = resolve_data_config(vars(args), model=model, verbose=False)
    loader_train, loader_eval, mixup_active, mixup_fn = eval('get_%s_data' % args.dataset)(
        batch_size=args.batch_size,
        step=args.step,
        dvs_da=args.DVS_DA,
        args=args,
        _logge=_logger,
        data_config=data_config,
        num_aug_splits=num_aug_splits,
        size=args.event_size,
        mix_up=args.mix_up,
        cut_mix=args.cut_mix,
        event_mix=args.event_mix,
        beta=args.cutmix_beta,
        prob=args.cutmix_prob,
        gaussian_n=args.gaussian_n,
        num=args.cutmix_num,
        noise=args.cutmix_noise,
        num_classes=args.num_classes,
        rand_aug=args.rand_aug,
        randaug_n=args.randaug_n,
        randaug_m=args.randaug_m,
        portion=args.train_portion,
        reconstruct=args.reconstructed,
        _logger=_logger,
        train_data_ratio=args.traindata_ratio,
        data_mode="full",
        frames_num=12,
        data_type="frequency"
    )

    source_loader_train, source_loader_list, _, _ = eval('get_transfer_%s_data' % args.dataset_two)(
        batch_size=args.batch_size,
        step=args.step,
        args=args,
        _logge=_logger,
        data_config=data_config,
        num_aug_splits=num_aug_splits,
        size=args.event_size,
        mix_up=args.mix_up,
        cut_mix=args.cut_mix,
        event_mix=args.event_mix,
        beta=args.cutmix_beta,
        prob=args.cutmix_prob,
        gaussian_n=args.gaussian_n,
        num=args.cutmix_num,
        noise=args.cutmix_noise,
        num_classes=args.num_classes,
        rand_aug=args.rand_aug,
        randaug_n=args.randaug_n,
        randaug_m=args.randaug_m,
        portion=args.train_portion,
        _logger=_logger,
        no_use_hsv=args.no_use_hsv
    )

    global CALTECH101_list
    global CEPDVS_list
    CALTECH101_list = source_loader_list
    CEPDVS_list = source_loader_list
    if args.loss_fn == 'mse':
        train_loss_fn = UnilateralMse(1.)
        validate_loss_fn = UnilateralMse(1.)

    else:
        if args.jsd:
            assert num_aug_splits > 1  # JSD only valid with aug splits set
            train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
        elif mixup_active:
            # smoothing is handled with mixup target transform
            train_loss_fn = SoftTargetCrossEntropy().cuda()
        elif args.smoothing:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
        else:
            train_loss_fn = nn.CrossEntropyLoss().cuda()

        validate_loss_fn = nn.CrossEntropyLoss().cuda()

    if args.loss_fn == 'mix':
        train_loss_fn = MixLoss(train_loss_fn)
        validate_loss_fn = MixLoss(validate_loss_fn)

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    from copy import deepcopy
    model_a = deepcopy(model)
    model_b = deepcopy(model)
    if args.eval:  # evaluate the model
        if args.distributed:
            state_dict = torch.load(args.eval_checkpoint)['state_dict_ema']
            new_state_dict = OrderedDict()
            # add module prefix for DDP
            for k, v in state_dict.items():
                k = 'module.' + k
                new_state_dict[k] = v

            model.load_state_dict(new_state_dict)
        else:
            if args.dataset == "dvsc10":
                model_a.load_state_dict(torch.load("/home/hexiang/TransferLearning_For_DVS/Results_lastest/Baseline/VGG_SNN-dvsc10-10-seed_42-bs_120-DA_True-ls_0.0-lr_0.005-traindataratio_1.0-TET_first_True-TET_second_True/model_best.pth.tar", map_location=torch.device('cpu'))['state_dict'])
            elif args.dataset == "NCALTECH101":
                model_a.load_state_dict(torch.load(
                    "/home/hexiang/DomainAdaptation_DVS/Results/Baseline/VGG_SNN-NCALTECH101-10-seed_42-bs_120-DA_False-ls_0.0-lr_0.005-traindataratio_1.0-TET_loss_True-refined_False/model_best.pth.tar",
                    map_location=torch.device('cpu'))['state_dict'])
            elif args.dataset == "omni":
                model_a.load_state_dict(torch.load(
                    "/home/hexiang/TransferLearning_For_DVS/Results_711/Baseline/SCNN-omni-12-seed_42-bs_64-DA_False-ls_0.0-lr_0.1-traindataratio_1.0-TET_first_False-TET_second_False-refined_False/model_best.pth.tar",
                    map_location=torch.device('cpu'))['state_dict'])
            else:
                pass
            if args.dataset_two == "dvsc10":
                model_b.load_state_dict(torch.load(
                    "/home/hexiang/TransferLearning_For_DVS/Results_lastest/Baseline/VGG_SNN-dvsc10-10-seed_47-bs_120-DA_True-ls_0.0-lr_0.005-traindataratio_1.0-TET_first_True-TET_second_True/model_best.pth.tar",
                    map_location=torch.device('cpu'))['state_dict'])
                # model_b.load_state_dict(torch.load(
                #     "/home/hexiang/TransferLearning_For_DVS/Results_additional/Baseline/VGG_SNN-cifar10-10-seed_42-bs_120-DA_True-ls_0.0-lr_0.005-traindataratio_1.0-TET_first_True-TET_second_True-refined_False/model_best.pth.tar",
                #     map_location=torch.device('cpu'))['state_dict'])
            elif args.dataset_two == "cifar10":
                model_b.load_state_dict(torch.load("/home/hexiang/TransferLearning_For_DVS/Results_additional/Baseline/VGG_SNN-cifar10-10-seed_42-bs_120-DA_True-ls_0.0-lr_0.005-traindataratio_1.0-TET_first_True-TET_second_True-refined_False/model_best.pth.tar", map_location=torch.device('cpu'))['state_dict'])
            elif args.dataset_two == "CALTECH101":
                model_b.load_state_dict(torch.load(
                    "/home/hexiang/DomainAdaptation_DVS/Results/Baseline/VGG_SNN-CALTECH101-10-seed_42-bs_120-DA_False-ls_0.0-lr_0.005-traindataratio_1.0-TET_loss_True-refined_False//model_best.pth.tar",
                    map_location=torch.device('cpu'))['state_dict'])
            else:
                pass
        for i in range(1):
            validate(start_epoch, model_a, model_b, loader_train, source_loader_train, validate_loss_fn, args,
                                   visualize=args.visualize, spike_rate=args.spike_rate,
                                   tsne=args.tsne, conf_mat=args.conf_mat)
        return



features_out_hook = []
def hook(module, fea_in, fea_out):
    global features_out_hook
    features_out_hook.append(fea_out)
    return None


def validate(epoch, model_a, model_b, loader, loader_two, loss_fn, args, amp_autocast=suppress,
             log_suffix='', visualize=False, spike_rate=False, tsne=False, conf_mat=False):
    # model_a 跑dvs, model_b 跑rgb
    model_a.eval()
    model_b.eval()
    last_idx = len(loader) - 1

    global features_out_hook
    features_a_out_hook = []
    features_b_out_hook = []
    for child in model_a.modules():
        if isinstance(child, nn.Conv2d) or isinstance(child, nn.BatchNorm2d) or isinstance(child, nn.Flatten) \
                or isinstance(child, nn.Linear) or isinstance(child, LIFNode) or isinstance(child, nn.AvgPool2d):
            child.register_forward_hook(hook=hook)

    for child in model_b.modules():
        if isinstance(child, nn.Conv2d) or isinstance(child, nn.BatchNorm2d) or isinstance(child, nn.Flatten) \
                or isinstance(child, nn.Linear) or isinstance(child, LIFNode) or isinstance(child, nn.AvgPool2d):
            child.register_forward_hook(hook=hook)

    with torch.no_grad():
        repeat = 16 # repeat 的次数是需要的sample数目除以batch size
        cka = torch.zeros((30, 30))
        scaled_hsic = torch.zeros(repeat, 30, 30)
        normalization_d = torch.zeros(repeat, 30, 30)
        source_input_list, source_label_list = next(iter(loader_two))
        global CALTECH101_list, CEPDVS_list

        for batch_idx, (inputs, target) in enumerate(tqdm(loader)):
            rgb_index = []
            for i in range(len(target)):
                import random
                rgb_index.append(random.randint(CALTECH101_list[target[i].item()][0], CALTECH101_list[target[i].item()][1]))
            inputs_two = source_input_list[rgb_index]
            if args.dataset_two == "cifar10" or args.dataset_two == "CALTECH101":
                inputs_two = inputs_two[:, 0:2, :, :]
            if batch_idx == repeat:
                break
            features_a_out_hook = [0.0 for i in range(0, 30)]
            features_b_out_hook = [0.0 for i in range(0, 30)]
            # inputs = inputs.type(torch.float64)
            last_batch = batch_idx == last_idx
            if not args.prefetcher or args.dataset != 'imnet':
                inputs = inputs.type(torch.FloatTensor).cuda()
                target = target.cuda()
                inputs_two = inputs_two.type(torch.FloatTensor).cuda()

            with amp_autocast():
                with torch.no_grad():
                    output_a = model_a(inputs)
                    margin = len(features_out_hook) // args.step
                    for i in range(0, margin):
                        for j in range(0, args.step):
                            features_a_out_hook[i] += features_out_hook[i + j * margin]
                        features_a_out_hook[i] /= args.step
                    features_out_hook = []

            with amp_autocast():
                with torch.no_grad():
                    output_b = model_b(inputs_two)  # inputs_two
                    features_b_out_hook = [0.0 for i in range(0, 30)]
                    margin = len(features_out_hook) // args.step
                    for i in range(0, margin):
                        for j in range(0, args.step):
                            features_b_out_hook[i] += features_out_hook[i + j * margin]
                        features_b_out_hook[i] /= args.step
                    features_out_hook = []


            if False:
                # 将Tensor转换为numpy数组，以便用matplotlib绘制
                tensor_a = model_a.feature[0].node.mem.cpu().reshape(-1).numpy()
                tensor_b = model_b.feature[0].node.mem.cpu().reshape(-1).numpy()

                # 使用scipy's gaussian_kde来估计PDF
                kde_a = stats.gaussian_kde(tensor_a)
                kde_b = stats.gaussian_kde(tensor_b)

                # 定义要在其上绘制PDF的值的范围
                len_va = 4 * 64 * 48 * 48 // 100
                x_a = np.linspace(tensor_a.min(), tensor_a.max(), len_va)
                x_b = np.linspace(tensor_b.min(), tensor_b.max(), len_va)

                # 计算每个值的PDF
                print("calculate pdf")
                pdf_a = kde_a.evaluate(x_a)
                pdf_b = kde_b.evaluate(x_b)

                # 使用matplotlib的plot函数绘制PDF
                plt.bar(x_a, pdf_a, width=(tensor_a.max() - tensor_a.min()) / float(len_va))
                plt.xlabel('Membrane potential value', fontsize=18)
                plt.ylabel('Density', fontsize=18)
                # plt.title('Distribution of membrane potential trained on DVS')
                plt.grid(True)
                plt.savefig('mem_caltech101.svg', dpi=600)

                plt.figure()

                # 使用matplotlib的plot函数绘制PDF
                plt.bar(x_b, pdf_b, width=(tensor_b.max() - tensor_b.min()) / float(len_va))
                plt.xlabel('Membrane potential value', fontsize=18)
                plt.ylabel('Density', fontsize=18)
                # plt.title('Distribution of membrane potential trained on RGB')
                plt.grid(True)
                plt.savefig('mem_ncaltech101.svg', dpi=600)
                print("mem plot finished!")
                sys.exit()

            for i in range(0, 30):
                for j in range(0, 30):
                    tmp_a, tmp_b = \
                        CKA.linear_CKA(features_a_out_hook[i].view(args.batch_size, -1),
                                       features_b_out_hook[j].view(args.batch_size, -1))
                    scaled_hsic[batch_idx][i][j] = tmp_a.cpu()
                    normalization_d[batch_idx][i][j] = tmp_b.cpu()

            torch.cuda.empty_cache()

        with amp_autocast():
            # features_a_used = [0.0 for i in range(0, 30)]
            # features_b_used = [0.0 for i in range(0, 30)]

            # for i in range(0, 30):
            #     tmp_list = []
            #     for j in range(0, repeat):
            #         tmp_list.append(features_a_out[j][i])
            #     features_a_used[i] = torch.cat(tmp_list, dim=0)
            #
            # for i in range(0, 30):
            #     tmp_list = []
            #     for j in range(0, repeat):
            #         tmp_list.append(features_b_out[j][i])
            #     features_b_used[i] = torch.cat(tmp_list, dim=0)

            cka = torch.zeros((30, 30))
            for i in range(0, 30):
                for j in range(0, 30):
                    cka[i][j] = torch.sum(scaled_hsic, dim=0)[i][j] / (torch.sum(normalization_d, dim=0)[i][j] + 1e-7)
            torch.save(cka, "./mycka_{}_{}.pt".format(args.dataset, args.dataset_two))
    return None


if __name__ == '__main__':
    torch.set_num_threads(20)
    os.environ["OMP_NUM_THREADS"] = "20"  # 设置OpenMP计算库的线程数
    os.environ["MKL_NUM_THREADS"] = "20"  # 设置MKL-DNN CPU加速库的线程数。
    # main()

    if True:
        sns.set_context("notebook", font_scale=1,
                        rc={"lines.linewidth": 2.5})

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        color_lib = sns.color_palette()

        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 16
        fig, ax = plt.subplots(figsize=(10, 6))

        # data = np.random.rand(20,20)
        data = torch.load("./mycka_NCALTECH101_CALTECH101.pt")
        # data = torch.load("./mycka_dvsc10_cifar10.pt")
        cka_diagonal = []
        for i in range(0, 30):
            cka_diagonal.append(data[i][i])

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_xlim([0, 30])

        h = sns.heatmap(data, cmap='inferno', xticklabels=5, yticklabels=5, cbar=False, vmin=0, vmax=1)
        h.invert_yaxis()
        cb=h.figure.colorbar(h.collections[0]) #显示colorbar
        cb.ax.tick_params(labelsize=16) #设置colorbar刻度字体大小
        # cb.set_label('colorbar')
        cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # cb.locator = matplotlib.ticker.MaxNLocator(nbins=6)
        cb.update_ticks()
        ax.tick_params(bottom=False, top=False, left=False, right=False)

        plt.xlabel('DVS trained SNN',fontsize=18, color='k') #x轴label的文本和字体大小
        args, args_text = _parse_args()
        if args.dataset_two == "dvsc10":
            plt.ylabel('DVS trained SNN',fontsize=18, color='k') #y轴label的文本和字体大小
        else:
            plt.ylabel('RGB trained SNN', fontsize=18, color='k')  # y轴label的文本和字体大小

        # plt.title('NCALTECH101',fontsize=20) #图片标题文本和字体大小

        plt.savefig('mycka_NCALTECH101_CALTECH101.svg', dpi=600)  # 图表输出
        # plt.savefig('mycka_dvsc10_cifar10.svg', dpi=600)  # 图表输出

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(0, 30), cka_diagonal, '*', linewidth=2, linestyle="dashdot")
        # ax.legend(bbox_to_anchor=(1, 1), loc=1, fontsize=13)
        plt.xlabel('Layers', fontsize=18)
        plt.ylabel('Similarity', fontsize=18)
        ax.set_xlim([0, 29])
        plt.savefig('centercka_NCALTECH101_CALTECH101.svg', dpi=600)  # 图表输出