#!/usr/bin/env/ python
# -*- coding: utf-8 -*-
""" 
@author:yanms 
@file: minist_classfication.py 
@time: 2020/11/13 
@version: V 0.1
@desc: 手写数字识别
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np

from visdom import Visdom

BATH_SIZE = 200
RATE = 0.01
EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("../data", train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATH_SIZE, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("../data", train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATH_SIZE, shuffle=True
)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x


viz = Visdom()

viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss&acc.', legend=['loss', 'acc.']))

net = Net().to(device)
optimizer = optim.SGD(net.parameters(), lr=RATE)
criterion = nn.CrossEntropyLoss().to(device)

global_step = 0

for epoch in range(EPOCHS):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.to(device)

        logistic = net(data)
        loss = criterion(logistic, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1
        viz.line([loss.item()], [global_step], win='train_loss', update='append')

        if batch_idx % 100 == 0:
            print("Train Epoch:{} [{}/{} ({:.0f}%)]\t Loss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()
            ))
    test_loss = 0.
    correct = 0.
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.to(device)

        logistic = net(data)
        test_loss += criterion(logistic, target).item()

        pred = logistic.argmax(dim=1)
        correct += pred.eq(target).float().sum().item()

    viz.line([[test_loss, correct / len(test_loader.dataset)]],
             [global_step], win='test', update='append')
    viz.images(data.view(-1, 1, 28, 28)*0.3081+0.1307, win='x')
    viz.text(str(pred.cpu().detach().numpy()), win='pred', opts=dict(title='pred'))

    test_loss /= len(test_loader.dataset)
    print("Test set: Average loss: {:.4f}, Accuracy: {}/{} {:.0f}%".format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)
    ))


# if __name__ == '__main__':
#     data = test_loader.dataset.data
#     target = test_loader.dataset.targets
#     data1 = data.view(-1, 1, 28, 28)
#     print(data.size())
#     print(data)
#     x = torch.full((28, 28), 0.4545).unsqueeze(0)
#     d1 = data1[:200]+x
#     print(d1)
#     print(data1[:200].size())
#     viz.images(data1[:200], win='x')
#     viz.text(str(target[:200].numpy()), win='pred', opts=dict(title='pred'))
#
#     print(x.size())

