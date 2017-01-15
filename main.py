import os
import argparse
from collections import OrderedDict
import json
import numpy as np
from chainer.functions.loss.softmax_cross_entropy import softmax_cross_entropy
from chainer.functions.evaluation.accuracy import accuracy
from chainer import datasets, Variable
from chainer.dataset import convert
import chainer

from resnet import ResNet
from train_cifar10 import train_cifar10

def main():
    args = parse_args()
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# iter: {}'.format(args.iter))
    print('# model: ResNet{}'.format(args.n*6 + 2))
    print('')

    model = ResNet(args.n)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    reports = train_cifar10(model, batchsize=args.batchsize, iter_=args.iter*1000, device=args.gpu, warm_up=args.warmup, verbose=True)

    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    with open("{}/log.json".format(args.out), "w") as f:
        json.dump(reports, f)

def parse_args():
    parser = argparse.ArgumentParser(description='resnet-cifart10')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--iter', '-i', type=int, default=64,
                        help='Number of iteration during training')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default=None,
                        help='Directory to output the result')
    parser.add_argument('--n', '-n', type=int, default=3,
                        help='The model will have 6n+2 layers')
    parser.add_argument('--warmup', '-w', action='store_true',
                        help='Warm up with lr=0.01')
    args = parser.parse_args()
    if args.out is None:
        args.out = "result_resnet{}".format(args.n*6 + 2)
    return args

if __name__ =="__main__":
    main()
