import sys
import mxnet as mx
from mxnet.gluon import data as gdata
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn

from gluonnlp.optimizer import LAMB


def test_lamb_for_fashion_mnist():
    mnist_train = gdata.vision.FashionMNIST(train=True)
    mnist_test = gdata.vision.FashionMNIST(train=False)

    batch_size = 512
    transformer = gdata.vision.transforms.ToTensor()
    if sys.platform.startswith('win'):
        num_workers = 0  # 0 disables multi-processing.
    else:
        num_workers = 4

    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                                  batch_size, shuffle=True,
                                  num_workers=num_workers)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                                 batch_size, shuffle=False,
                                 num_workers=num_workers)

    net = nn.Sequential()
    net.add(nn.Conv2D(6, kernel_size=5),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(16, kernel_size=5),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Dense(120),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Dense(84),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Dense(10))

    ctx = mx.cpu()
    net.initialize(ctx=ctx)

    trainer = gluon.Trainer(net.collect_params(), 'LAMB', {'learning_rate': 0.001})

    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    num_epochs = 5

    def evaluate_accuracy(data_iter, net, ctx):
        """Evaluate accuracy of a model on the given data set."""
        acc_sum, n = 0.0, 0.0
        for X, y in train_iter:
            X = X.as_in_context(ctx)
            y = y.as_in_context(ctx)
            y_hat = net(X)

            y = y.astype('float32')
            acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        return acc_sum / n

    def train(net, train_iter, test_iter, loss, num_epochs, batch_size,
              trainer, ctx):
        for epoch in range(num_epochs):
            train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
            for X, y in train_iter:
                X = X.as_in_context(ctx)
                y = y.as_in_context(ctx)
                with autograd.record():
                    y_hat = net(X)
                    l = loss(y_hat, y).sum()
                l.backward()

                trainer.step(batch_size)
                y = y.astype('float32')
                train_l_sum += l.asscalar()
                train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
                n += y.size
            test_acc = evaluate_accuracy(test_iter, net, ctx)
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
                  % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

    train(net, train_iter, test_iter, loss, num_epochs, batch_size, trainer, ctx)
