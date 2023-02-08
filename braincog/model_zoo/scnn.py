# -*- coding: utf-8 -*-            
# Time : 2023/2/8 15:03
# Author : Regulus
# FileName: scnn.py
# Explain:

from functools import partial
from torch.nn import functional as F
import torchvision
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule, BaseLinearModule
from braincog.datasets import is_dvs_data


@register_model
class SCNN(BaseModule):
    def __init__(self,
                 num_classes=1623,
                 step=12,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.n_preact = kwargs['n_preact'] if 'n_preact' in kwargs else False

        self.num_classes = num_classes

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']
        self.feature = nn.Sequential(
            BaseConvModule(2, 15, kernel_size=(5, 5), padding=(0, 0), node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(15, 40, kernel_size=(5, 5), padding=(0, 0), node=self.node),
            nn.AvgPool2d(2),
        )

        self.fc1 = nn.Linear(640, 300)
        self.node1 = self.node()
        self.fc2 = nn.Linear(300, 1623)

    def forward(self, inputs):
        inputs = self.encoder(inputs)
        self.reset()

        output = 0.0
        for t in range(self.step):
            # add encode output to list (firing rate)
            x = inputs[t]

            # add feature output to list (membrane potential)
            x = self.feature(x)

            x = x.view(x.shape[0], -1)
            x = self.fc1(x)
            x = self.node1(x)
            output += self.fc2(x)
        output /= self.step
        return output


@register_model
class Transfer_SCNN(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.n_preact = kwargs['n_preact'] if 'n_preact' in kwargs else False

        self.num_classes = num_classes

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']

        self.feature = nn.Sequential(
            BaseConvModule(2, 15, kernel_size=(5, 5), padding=(0, 0), node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(15, 40, kernel_size=(5, 5), padding=(0, 0), node=self.node),
            nn.AvgPool2d(2),
        )

        self.fc1 = nn.Linear(640, 300)
        self.node1 = self.node()
        self.fc2 = nn.Linear(300, 1623)


    def forward(self, inputs_rgb, inputs_dvs):
        inputs_rgb = self.encoder(inputs_rgb)
        inputs_dvs = self.encoder(inputs_dvs)
        self.reset()

        outputs_rgb_feature, outputs_dvs_feature = [], []
        outputs_rgb, outputs_dvs = [], []
        for t in range(self.step):
            # add encode output to list (firing rate)
            x_rgb = inputs_rgb[t]

            # add feature output to list (membrane potential)
            x_rgb = self.feature(x_rgb)
            outputs_rgb_feature.append(self.feature[-2].node.mem)

            # add fc output to list (firing rate)
            x_rgb = x_rgb.view(x_rgb.shape[0], -1)
            x_rgb = self.fc1(x_rgb)
            x_rgb = self.node1(x_rgb)
            x_rgb = self.fc2(x_rgb)
            outputs_rgb.append(x_rgb)

        self.reset()
        for t in range(self.step):
            x_dvs = inputs_dvs[t]
            x_dvs = self.feature(x_dvs)
            outputs_dvs_feature.append(self.feature[-2].node.mem)
            x_dvs = x_dvs.view(x_dvs.shape[0], -1)
            x_dvs = self.fc1(x_dvs)
            x_dvs = self.node1(x_dvs)
            x_dvs = self.fc2(x_dvs)
            outputs_dvs.append(x_dvs)

        return outputs_rgb_feature, outputs_dvs_feature, outputs_rgb, outputs_dvs