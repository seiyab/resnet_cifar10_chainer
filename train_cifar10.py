import os
import argparse
from collections import OrderedDict
import json
import numpy as np
import cupy as cp
from chainer.functions.loss.softmax_cross_entropy import softmax_cross_entropy
from chainer.functions.evaluation.accuracy import accuracy
from chainer import datasets, Variable
from chainer.dataset import convert
import chainer

def train_cifar10(model, batchsize=128, iter_=64000, device=-1, warm_up=False, verbose=False):
    optimizer = chainer.optimizers.MomentumSGD(lr=0.1)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    train_raw, test_raw = datasets.get_cifar10()
    train_images_raw, train_labels = convert.concat_examples(train_raw, device)
    test_images_raw, test_labels = convert.concat_examples(test_raw, device)
    train_images, test_images = pixwise_normalize(train_images_raw, test_images_raw, scale=False)
    train = datasets.tuple_dataset.TupleDataset(train_images, train_labels)
    test = datasets.tuple_dataset.TupleDataset(test_images, test_labels)

    train_iter = chainer.iterators.SerialIterator(train, batchsize)

    reports = []
    if not warm_up:
        lr_schedule = [(0, 0.1), (32000, 0.01), (48000, 0.001), (-1,)]
    else:
        lr_schedule = [(0, 0.01), (400, 0.1), (32000, 0.01), (48000, 0.001), (-1,)]
    lr_change = lr_schedule.pop(0)
    for i, batch in enumerate(train_iter):
        if i==lr_change[0]:
            optimizer.lr = lr_change[1]
            lr_change = lr_schedule.pop(0)
        if train_iter.is_new_epoch or i==0:
            test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)
            reports.append(report(model, {"train":[batch], "test":test_iter}, i, train_iter.epoch, verbose=True, device=device))

        images_raw, labels = convert.concat_examples(batch, device)
        images = flip(crop(images_raw, device=device))
        optimizer.update(lambda x, t:softmax_cross_entropy(model(x), t), *(Variable(images), Variable(labels)))

        if i == iter_:
            test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)
            reports.append(report(model, {"train":[batch], "test":test_iter}, i, train_iter.epoch, verbose=True, device=device))
            break

    return reports

def report(model, data, iter_num, epoch_num, verbose=False, device=-1):
    d = [("iter", iter_num), ("epoch_num", epoch_num)]
    for name, batches in data.items():
        arrays = [(convert.concat_examples(batch, device), len(batch)) for batch in batches]
        arrays, lens = map(list, zip(*arrays))
        data_size = sum(lens)
        losses = [
            float((lambda x, t:softmax_cross_entropy(model(x, test=True), t))(*map(Variable, array)).data) * len(array[0])
            for array in arrays
        ]
        accuracies = [
            float((lambda x, t:accuracy(model(x, test=True), t))(*map(Variable, array)).data) * len(array[0])
            for array in arrays
        ]
        d.append(("{}/loss".format(name), sum(losses) / data_size))
        d.append(("{}/error".format(name), 1 - sum(accuracies) / data_size))
    if verbose:
        for name, value in d:
            print("{}: {}".format(name, value))
        print()
    return OrderedDict(d)


def crop(x, padding=4, device=-1):
    xp = cp if device>=0 else np
    padded = xp.zeros(np.array(x.shape) + np.array([0, 0, padding*2, padding*2]), dtype=np.float32)
    padded[:,:,padding:-padding,padding:-padding] = x
    positions = np.random.randint(padding*2, size=2)
    return padded[:,:,positions[0]:positions[0]+x.shape[-2],positions[1]:positions[1]+x.shape[-1]]

def flip(x, p=0.5):
    return x[:,:,:,::-1] if np.random.binomial(1, p) == 1 else x

def pixwise_normalize(train, test, scale=True):
    avg = train.mean(axis=0)
    if scale:
        raise NotImplementedError
    else:
        return train - avg, test - avg
