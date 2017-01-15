# resnet_cifar10
train ResNet using chainer.
# test error(%)
model    |my code|the original paper
---------|------:|-----------------:
ResNet20 |   8.87|              8.75
ResNet32 |   7.23|              7.51
ResNet44 |   6.81|              7.17
ResNet56 |   7.92|              6.97
ResNet110|   6.52|              6.61
Except ResNet56, my implementation reproduced test error.
However, ResNet56 was much poorer than original.
I have no idea for the reason.
