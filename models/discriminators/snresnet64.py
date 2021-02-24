import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import utils

from models.discriminators.resblocks import Block
from models.discriminators.resblocks import OptimizedBlock


class Omniglot_Discriminator(nn.Module):

    def __init__(self, num_features=32, num_classes=0, activation=F.relu):
        super(Omniglot_Discriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(1, num_features)
        self.block2 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True)
        self.block3 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)
        self.l4 = utils.spectral_norm(nn.Linear(num_features * 4, 1))

        self.linear_mi = nn.Linear(num_features * 4, num_classes)
        self.linear_c = nn.Linear(num_features * 4, num_classes)

        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                nn.Embedding(num_classes, num_features * 4))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l4.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l4(h)

        out_mi = self.linear_mi(h)
        out_c = self.linear_c(h)

        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output, out_mi, out_c


class VGG_Discriminator(nn.Module):

    def __init__(self, num_features=64, num_classes=0, activation=F.relu):
        super(VGG_Discriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features // 4)
        self.block2 = Block(num_features // 4, num_features // 2,
                            activation=activation, downsample=True)
        self.block3 = Block(num_features // 2, num_features,
                            activation=activation, downsample=True)
        self.block4 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True)
        self.block5 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)
        self.l4 = utils.spectral_norm(nn.Linear(num_features * 4, 1))

        self.linear_mi = nn.Linear(num_features * 4, num_classes)
        self.linear_c = nn.Linear(num_features * 4, num_classes)

        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                nn.Embedding(num_classes, num_features * 4))
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l4.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l4(h)

        out_mi = self.linear_mi(h)
        out_c = self.linear_c(h)

        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output, out_mi, out_c
