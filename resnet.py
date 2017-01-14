from collections import OrderedDict
import inspect
import numpy as np

from chainer import Variable
from chainer.link import Chain
from chainer.links.connection.convolution_2d import Convolution2D
from chainer.links.normalization.batch_normalization import BatchNormalization
from chainer.links.connection.linear import Linear
from chainer.functions.activation.relu import relu
from chainer.functions import concat
import chainer.functions as F
from chainer.functions.pooling.max_pooling_2d import max_pooling_2d
from chainer.functions.pooling.average_pooling_2d import average_pooling_2d
from chainer.initializers import normal
from chainer.links.model.vision.resnet import _global_average_pooling_2d

class ResNet(Chain):
    def __init__(self, n, n_channels=[16, 32, 64]):
        kwargs = {'initialW': normal.HeNormal(scale=1.0)}
        super(ResNet, self).__init__(
            conv1=Convolution2D(3, n_channels[0], 3, 1, 1, nobias=True, **kwargs),
            bn1=BatchNormalization(n_channels[0]),
            res2=Group(n_channels[0], n, **kwargs),
            sub3=PaddingShortcutBlock(*n_channels[:2], SubsamplingBlock(*n_channels[:2], **kwargs), **kwargs),
            res4=Group(n_channels[1], n-1, **kwargs),
            sub5=PaddingShortcutBlock(*n_channels[1:], SubsamplingBlock(*n_channels[1:], **kwargs), **kwargs),
            res6=Group(n_channels[2], n-1, **kwargs),
            fc7=Linear(n_channels[-1], 10),
        )
        self.functions = OrderedDict([
            ('conv1', [self.conv1, self.bn1, relu]),
            ('group2', [self.res2]),
            ('group3', [self.sub3, self.res4]),
            ('groop4', [self.sub5, self.res6]),
            ('pool4', [_global_average_pooling_2d]),
            ('fc5', [self.fc7]),
        ])

    def __call__(self, x, test=False):
        h = x
        for key, funcs in self.functions.items():
            for func in funcs:
                if "test" in inspect.getargspec(func)[0]:
                    h = func(h, test=test)
                else:
                    h = func(h)
        return h


class IdentityShortcutBlock(Chain):
    def __init__(self, main_path):
        super(IdentityShortcutBlock, self).__init__(main_path=main_path)

    def __call__(self, x, test=False):
        if "test" in inspect.getargspec(self.main_path)[0]:
            h = self.main_path(x, test=test)
        else:
            h = self.main_path(x)
        return relu(h + x)

class PaddingShortcutBlock(Chain):
    def __init__(self, in_channels, out_channels, main_path, stride=2, initialW=None):
        self.padding_channels = out_channels - in_channels
        super(PaddingShortcutBlock, self).__init__(
            main_path=main_path
        )

    def __call__(self, x, test=False):
        if "test" in inspect.getargspec(self.main_path)[0]:
            h = self.main_path(x, test=test)
        else:
            h = self.main_path(x)
        shortcut = average_pooling_2d(x, 2)
        zeros = Variable(self.xp.zeros((shortcut.shape[0], self.padding_channels, *shortcut.shape[2:]), dtype=self.xp.float32))
        shortcut = F.concat((shortcut, zeros))
        return relu(h + shortcut)

class ProjectionShortcutBlock(Chain):
    def __init__(self, in_channels, out_channels, main_path, stride=2, initialW=None):
        super(ProjectionShortcutBlock, self).__init__(
            shortcut=Convolution2D(
                in_channels, out_channels, 1, stride, 0,
                initialW=initialW, nobias=True),
            bn = BatchNormalization(out_channels),
            main_path=main_path
        )

    def __call__(self, x, test=False):
        if "test" in inspect.getargspec(self.main_path)[0]:
            h = self.main_path(x, test=test)
        else:
            h = self.main_path(x)
        return relu(h + self.bn(self.shortcut(x), test=test))

class BasicBlock(Chain):
    def __init__(self, n_channel, initialW=None):
        super(BasicBlock, self).__init__(
            conv1=Convolution2D(
                n_channel, n_channel, 3, 1, 1,
                initialW=initialW, nobias=True
            ),
            bn1=BatchNormalization(n_channel),
            conv2=Convolution2D(
                n_channel, n_channel, 3, 1, 1,
                initialW=initialW, nobias=True
            ),
            bn2=BatchNormalization(n_channel)
        )

    def __call__(self, x, test=False):
        h = relu(self.bn1(self.conv1(x), test=test))
        return self.bn2(self.conv2(h), test=test)

class SubsamplingBlock(Chain):
    def __init__(self, in_channels, out_channels, stride=2, initialW=None):
        super(SubsamplingBlock, self).__init__(
            conv1=Convolution2D(
                in_channels, out_channels, 3, stride, 1,
                initialW=initialW, nobias=True
            ),
            bn1=BatchNormalization(out_channels),
            conv2=Convolution2D(
                out_channels, out_channels, 3, 1, 1,
                initialW=initialW, nobias=True
            ),
            bn2=BatchNormalization(out_channels)
        )

    def __call__(self, x, test=False):
        h = relu(self.bn1(self.conv1(x), test=test))
        return self.bn2(self.conv2(h), test=test)


class Group(Chain):
    def __init__(self, n_channel, n_layer, initialW=None):
        self.layers = [IdentityShortcutBlock(BasicBlock(n_channel, initialW=initialW)) for _ in range(n_layer)]
        super(Group, self).__init__(
                **{"layer{}".format(i): layer for i, layer in enumerate(self.layers)}
        )

    def __call__(self, x, test=False):
        h = x
        for layer in self.layers:
            h = layer(h, test=test)
        return h
