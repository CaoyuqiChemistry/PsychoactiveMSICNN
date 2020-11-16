import torch
import copy
import time
import pandas as pd


def MyTrainModel(model,traindataloader,train_rate,criterion,optimizer,num_epochs=25):

    batch_num = len(traindataloader)
    train_batch_num = round(batch_num * train_rate)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*10)
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0
        val_corrects = 0
        val_num = 0
        for step,(b_x,b_y) in enumerate(traindataloader):
            if step < train_batch_num:
                model.train()
                output = model(b_x)
                pre_lab = torch.argmax(output,1)
                loss = criterion(output,b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * b_x.size(0)
                train_corrects += torch.sum(pre_lab == b_y.data)
                train_num += b_x.size(0)
            else:
                model.eval()
                output = model(b_x)
                pre_lab = torch.argmax(output,1)
                loss = criterion(output,b_y)
                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(pre_lab == b_y.data)
                val_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item()/train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item()/val_num)
        print('{} Train loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val loss: {:.4f} val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] >= best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            time_use = time.time() - since
            print('Train and val complete in {:.0f}m {:.0f}s'.format(time_use//60, time_use % 60))

    model.load_state_dict(best_model_wts)
    train_process = pd.DataFrame(
        data={
            "epoch" : range(num_epochs),
            "train_loss_all" : train_loss_all,
            "val_loss_all" : val_loss_all,
            "train_acc_all" : train_acc_all,
            "val_acc_all":val_acc_all,
        }
    )
    return model, train_process
