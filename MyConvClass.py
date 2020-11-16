# -*- coding: utf-8 -*-

import torch.nn as nn
from torchvision import models

vgg16 = models.vgg16(pretrained=True)
vgg = vgg16.features[1:]
for param in vgg.parameters():
    param.requires_grad_(False)

class MyConvClass(nn.Module):
    def __init__(self,input_c):
        super(MyConvClass,self).__init__()

        self.conv1 = nn.Conv2d(
                in_channels=input_c,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            )

        self.vgg = vgg

        self.classifier = nn.Sequential(
            nn.Linear(25088,512),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(256,5),
            nn.Softmax(dim = 1)
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.vgg(x)
        x = x.view(x.size(0),-1)
        output = self.classifier(x)
        return output
