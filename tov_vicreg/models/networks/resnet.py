from torch import nn
import numpy as np

def fixup_init(layer, num_layers):
    nn.init.normal_(layer.weight, mean=0, std=np.sqrt(
        2 / (layer.weight.shape[0] * np.prod(layer.weight.shape[2:]))) * num_layers ** (-0.25))


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio,
                 norm_type, num_layers=1, groups=-1,
                 drop_prob=0., bias=True):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2, 3]
        self.drop_prob = drop_prob

        hidden_dim = round(in_channels * expand_ratio)

        if groups <= 0:
            groups = hidden_dim

        conv = nn.Conv2d

        if stride != 1:
            self.downsample = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
            nn.init.normal_(self.downsample.weight, mean=0, std=
                            np.sqrt(2 / (self.downsample.weight.shape[0] *
                            np.prod(self.downsample.weight.shape[2:]))))
        else:
            self.downsample = False

        if expand_ratio == 1:
            conv1 = conv(hidden_dim, hidden_dim, 3, stride, 1, groups=groups, bias=bias)
            conv2 = conv(hidden_dim, out_channels, 1, 1, 0, bias=bias)
            fixup_init(conv1, num_layers)
            fixup_init(conv2, num_layers)
            self.conv = nn.Sequential(
                # dw
                conv1,
                init_normalization(hidden_dim, norm_type),
                nn.ReLU(inplace=True),
                # pw-linear
                conv2,
                init_normalization(out_channels, norm_type),
            )
            nn.init.constant_(self.conv[-1].weight, 0)
        else:
            conv1 = conv(in_channels, hidden_dim, 1, 1, 0, bias=bias)
            conv2 = conv(hidden_dim, hidden_dim, 3, stride, 1, groups=groups, bias=bias)
            conv3 = conv(hidden_dim, out_channels, 1, 1, 0, bias=bias)
            fixup_init(conv1, num_layers)
            fixup_init(conv2, num_layers)
            fixup_init(conv3, num_layers)
            self.conv = nn.Sequential(
                # pw
                conv1,
                init_normalization(hidden_dim, norm_type),
                nn.ReLU(inplace=True),
                # dw
                conv2,
                init_normalization(hidden_dim, norm_type),
                nn.ReLU(inplace=True),
                # pw-linear
                conv3,
                init_normalization(out_channels, norm_type)
            )
            if norm_type != "none":
                nn.init.constant_(self.conv[-1].weight, 0)

    def forward(self, x):
        if self.downsample:
            identity = self.downsample(x)
        else:
            identity = x
        if self.training and np.random.uniform() < self.drop_prob:
            return identity
        else:
            return identity + self.conv(x)

class ResnetCNN(nn.Module):
    def __init__(self, input_channels=3,
                 depths=[32, 64, 64],
                 strides=[3, 2, 2],
                 blocks_per_group=3,
                 norm_type="bn",
                 resblock=InvertedResidual,
                 expand_ratio=2):
        super(ResnetCNN, self).__init__()
        self.depths = [input_channels] + depths
        self.resblock = resblock
        self.expand_ratio = expand_ratio
        self.blocks_per_group = blocks_per_group
        self.layers = []
        self.norm_type = norm_type
        self.num_layers = self.blocks_per_group*len(depths)
        for i in range(len(depths)):
            self.layers.append(self._make_layer(self.depths[i],
                                                self.depths[i+1],
                                                strides[i],
                                                ))
        self.layers.append(nn.Flatten())
        self.layers = nn.Sequential(*self.layers)
        self.train()

    def _make_layer(self, in_channels, depth, stride,):

        blocks = [self.resblock(in_channels, depth,
                                expand_ratio=self.expand_ratio,
                                stride=stride,
                                norm_type=self.norm_type,
                                num_layers=self.num_layers,)]

        for i in range(1, self.blocks_per_group):
            blocks.append(self.resblock(depth, depth,
                                        expand_ratio=self.expand_ratio,
                                        stride=1,
                                        norm_type=self.norm_type,
                                        num_layers=self.num_layers,))

        return nn.Sequential(*blocks)

    @property
    def local_layer_depth(self):
        return self.depths[-2]

    def forward(self, inputs):
        return self.layers(inputs)
        
def init_normalization(channels, type="bn", affine=True, one_d=False):
    assert type in ["bn", "ln", "in", "gn", "max", "none", None]
    if type == "bn":
        if one_d:
            return nn.BatchNorm1d(channels, affine=affine)
        else:
            return nn.BatchNorm2d(channels, affine=affine)
    elif type == "ln":
        if one_d:
            return nn.LayerNorm(channels, elementwise_affine=affine)
        else:
            return nn.GroupNorm(1, channels, affine=affine)
    elif type == "in":
        return nn.GroupNorm(channels, channels, affine=affine)
    elif type == "gn":
        groups = max(min(32, channels//4), 1)
        return nn.GroupNorm(groups, channels, affine=affine)
    elif type == "none" or type is None:
        return nn.Identity()