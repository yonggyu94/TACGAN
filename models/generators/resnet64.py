import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from models.generators.resblocks import Block


class Omniglot_Generator(nn.Module):
    """Generator generates 64x64."""
    def __init__(self, num_features=32, dim_z=128, bottom_width=7,
                 activation=F.relu, num_classes=0, distribution='normal'):
        super(Omniglot_Generator, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes
        self.distribution = distribution

        self.l1 = nn.Linear(dim_z, 4 * num_features * bottom_width ** 2)

        self.block2 = Block(num_features * 4, num_features * 2,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block3 = Block(num_features * 2, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.b4 = nn.BatchNorm2d(num_features)
        self.conv4 = nn.Conv2d(num_features, 1, 1, 1)

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.tensor)
        init.xavier_uniform_(self.conv4.weight.tensor)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in range(2, 4):
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b4(h))
        return torch.tanh(self.conv4(h))


class VGG_Generator(nn.Module):
    """Generator generates 64x64."""
    def __init__(self, num_features=64, dim_z=128, bottom_width=4,
                 activation=F.relu, num_classes=0, distribution='normal'):
        super(VGG_Generator, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes
        self.distribution = distribution

        self.l1 = nn.Linear(dim_z, 4 * num_features * bottom_width ** 2)

        self.block2 = Block(num_features * 4, num_features * 2,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block3 = Block(num_features * 2, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block4 = Block(num_features, num_features // 2,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block5 = Block(num_features // 2, num_features // 4,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.b4 = nn.BatchNorm2d(num_features // 4)
        self.conv4 = nn.Conv2d(num_features // 4, 3, 1, 1)

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.tensor)
        init.xavier_uniform_(self.conv4.weight.tensor)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in range(2, 6):
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b4(h))
        return torch.tanh(self.conv4(h))
