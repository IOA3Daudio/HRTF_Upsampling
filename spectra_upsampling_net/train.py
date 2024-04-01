import os
import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data_processing import myDataset
from build_net import hrtf_net
import scipy.io as sio

## functions
def train(epoch,fold):
    net.train(True)
    train_loss = 0
    dataloader = train_dataloader
    for batch_idx, (x_data,y_data) in enumerate(dataloader):
        x_data = x_data.to(device)
        y_data = y_data.to(device)
        optimizer.zero_grad()
        out_x,out = net(x_data)
        loss_1 = lossF(out_x, x_data)
        loss_2 = lossF(out, y_data)
        loss = loss_1 + loss_2
        
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += loss.item()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                    epoch, batch_idx, len(dataloader),100. * batch_idx / len(dataloader), loss.item()))
    net.train(False)
    with torch.no_grad():
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(dataloader)))
        history['Train Loss'].append(train_loss / len(dataloader))
        if epoch >= int(EPOCHS*3/4) - 1:
            torch.save(net.state_dict(),'./model/'+str(fold)+'/net'+str(epoch)+'.pth')

def valid(epoch):
    net.train(False)
    valid_loss = 0
    dataloader = valid_dataloader
    with torch.no_grad():
        for (x_data,y_data) in dataloader:
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            out_x,out = net(x_data)
            loss_1 = lossF(out_x, x_data)
            loss_2 = lossF(out, y_data)
            loss = loss_1 + loss_2

            valid_loss += loss.item()
        print('====> Valid Average loss: {:.4f}'.format(valid_loss / len(dataloader)))  
        history['Valid Loss'].append(valid_loss / len(dataloader))





seed = 666
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device_ids = [3]
device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCHS = 800

## LOAD ORI DATA
hrtf_sparse = sio.loadmat('./data/hrtf_all_measured.mat')['hrtf_sparse']
hrtf_dense = sio.loadmat('./data/hrtf_all_measured.mat')['hrtf_dense']
print(hrtf_sparse.shape)
print(hrtf_dense.shape) 
hrtf_mean = np.mean(hrtf_dense)
hrtf_std = np.std(hrtf_dense)
print(hrtf_mean)
print(hrtf_std)
hrtf_dense = (hrtf_dense - hrtf_mean)/hrtf_std
hrtf_sparse = (hrtf_sparse - hrtf_mean)/hrtf_std

K = 9
subject_all = np.arange(96)
best_model_ind_kfolds = np.zeros((K,1))
for fold in range(K):
    valid_subject_ind = subject_all[fold*10:(fold+1)*10]
    train_subject_ind = np.setdiff1d(subject_all, valid_subject_ind)
    hrtf_train_sparse = hrtf_sparse[train_subject_ind,:,:,:]
    hrtf_valid_sparse = hrtf_sparse[valid_subject_ind,:,:,:]
    hrtf_train_dense = hrtf_dense[train_subject_ind,:,:,:]
    hrtf_valid_dense = hrtf_dense[valid_subject_ind,:,:,:]

    print('Load Train Data:')
    train_dataset = myDataset(hrtf_train_sparse,hrtf_train_dense)
    print('Load Valid Data:')
    valid_dataset = myDataset(hrtf_valid_sparse,hrtf_valid_dense)
    train_dataloader = DataLoader(dataset = train_dataset,batch_size = BATCH_SIZE,shuffle = True)
    valid_dataloader = DataLoader(dataset = valid_dataset,batch_size = BATCH_SIZE,shuffle = False)

    ## Initial NET
    net = hrtf_net()
    if len(device_ids) > 1:
        net = nn.DataParallel(net,device_ids=device_ids)
    net = net.to(device)
    lossF = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    history = {'Train Loss':[],'Valid Loss':[]}

    ## Train
    torch.cuda.empty_cache()
    for epoch in range(EPOCHS):
        train(epoch,fold)
        valid(epoch)

    ## plot loss
    net.train(False)
    plt.clf()
    plt.plot(history['Train Loss'][5:],label = 'Train Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(history['Valid Loss'][5:],label = 'Valid Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.show()
    plt.savefig('./Loss'+str(fold)+'.png')
    print(min(history['Train Loss']))
    print(min(history['Valid Loss']))
    best_model_ind = history['Valid Loss'].index(min(history['Valid Loss'][-int(EPOCHS/4):]))
    print(best_model_ind)
    print(history['Train Loss'][best_model_ind])
    print(history['Valid Loss'][best_model_ind])

    best_model_ind_kfolds[fold] = best_model_ind


np.save("best_model_ind_kfolds.npy",best_model_ind_kfolds)
