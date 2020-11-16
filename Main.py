import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import seaborn as sns
import copy
import time
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
from MyConvClass import MyConvClass
from MyDataSetClass import MyDataSetClass
from MyTrainModel import MyTrainModel
from torchvision import transforms

if __name__ == '__main__':

    train_data = MyDataSetClass(
        root = 'G:/PycharmProject/MyConvProject/TrainingData',
        t = transforms.ToTensor()
    )

    train_data_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=2,
        shuffle=True,
        num_workers=2
    )

    myconvnet = MyConvClass(15)
    optimizer = torch.optim.Adam(myconvnet.parameters(),lr = 0.0003)
    criterion = nn.CrossEntropyLoss()
    myconvnet, train_process = MyTrainModel(myconvnet,train_data_loader,0.7,criterion,optimizer,num_epochs=50)

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process.epoch,train_process.train_loss_all,'ro-',label = "Train loss")
    plt.plot(train_process.epoch,train_process.val_loss_all,"bs-", label = "Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1,2,2)
    plt.plot(train_process.epoch,train_process.train_acc_all,'ro-',label = "Train acc")
    plt.plot(train_process.epoch,train_process.val_acc_all,"bs-", label = "Val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.show()

    test_data = MyDataSetClass(
        root = 'G:/PycharmProject/MyConvProject/testdata',
        t = transforms.ToTensor()
    )

    b_x, b_y= [], []
    for step, (test_x,test_y) in enumerate(test_data):
        b_x.append(test_x.unsqueeze(0))
        b_y.append(test_y)

    b_x = torch.cat(b_x)
    b_y = torch.tensor(b_y)

    myconvnet.eval()
    output = myconvnet(b_x)
    prelab = torch.argmax(output,1)
    val_corrects = torch.sum(prelab == b_y.data)
    val_sum = b_x.size(0)
    acc = val_corrects.double().item() / val_sum
    print(acc)