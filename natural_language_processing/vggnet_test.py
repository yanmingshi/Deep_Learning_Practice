"""
_*_ coding: utf-8 _*_
@Time : 2020/11/3 19:01
@Author : yan_ming_shi
@Version：V 0.1
@File : vggnet_test.py
@desc :
"""

from __future__ import division
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
import os

import matplotlib.pyplot as plt

unloader = transforms.ToPILImage()  # reconvert into PIL image

LOG_DOR = "checkpoint/model.pth"


def imshow(tensor, title=None):
    """
    绘制图片
    :param tensor:
    :param title:
    :return:
    """
    image = tensor.clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.axis('off')  # 不显示坐标轴
    plt.imshow(image)
    plt.savefig('img/target_{}.jpg'.format(step / 10))
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def load_image(image_path, transform=None, max_size=None, shape=None):
    """
    读取图片 并进行预处理
    :param image_path: 图片路径
    :param transform:
    :param max_size:
    :param shape:
    :return:
    """
    image = Image.open(image_path)
    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale  # 将图片按最大长度比例进行缩放，满足最大边为max_size个长度
        image = image.resize(size.astype(int), Image.ANTIALIAS)

    if shape:
        image = image.resize(shape, Image.LANCZOS)

    if transform:
        image = transform(image).unsqueeze(0)

    return image


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])  # 来自ImageNet的mean和variance

content = load_image("026.jpg", transform, max_size=400)
style = load_image("style.jpg", transform, shape=[content.size(2), content.size(3)])


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


if __name__ == '__main__':


    total_step = 2000
    style_weight = 100.
    # 优化target图片，图片初始化内容和原图片一致
    target = content.clone().requires_grad_(True)
    vgg = VGGNet().eval()  # eval() 表示不会被优化
    target_features = vgg(target)
    optimizer = torch.optim.Adam([target], lr=0.003, betas=[0.5, 0.999])

    if os.path.exists(LOG_DOR):
        checkpoint = torch.load(LOG_DOR)
        target = checkpoint['target'].requires_grad_(True)
        style = checkpoint['style'].requires_grad_(True)
        optimizer = torch.optim.Adam([target], lr=0.003, betas=[0.5, 0.999])
        optimizer.load_state_dict(checkpoint['optimizer'])


    for step in range(total_step):

        target_features = vgg(target)
        content_features = vgg(content)
        style_features = vgg(style)

        style_loss = 0
        content_loss = 0
        for f1, f2, f3 in zip(target_features, content_features, style_features):
            content_loss += torch.mean((f1 - f2) ** 2)
            _, c, h, w = f1.size()
            f1 = f1.view(c, h * w)
            f3 = f3.view(c, h * w)

            # 计算gram matrix
            f1 = torch.mm(f1, f1.t())
            f3 = torch.mm(f3, f3.t())
            style_loss += torch.mean((f1 - f3) ** 2) / (c * h * w)

        loss = content_loss + style_weight * style_loss

        # 更新target
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if step % 10 == 0:
            print("Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}"
                  .format(step, total_step, content_loss.item(), style_loss.item()))
            torch.save({'target': target, 'style': style, 'optimizer': optimizer.state_dict()}, LOG_DOR)

            denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
            img = target.clone().squeeze()
            img = denorm(img).clamp_(0, 1)
            plt.figure()

            imshow(img)
