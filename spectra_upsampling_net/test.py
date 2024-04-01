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
def test(dataloader):
    net.eval()
    data_true = []
    data_true_sparse = []
    data_pred = []
    data_recon = []
    with torch.no_grad():
        for (x_data,y_data) in dataloader:
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            out_x,out = net(x_data)

            data_true.extend(y_data.cpu().detach().numpy().tolist())
            data_pred.extend(out.cpu().detach().numpy().tolist())
            data_recon.extend(out_x.cpu().detach().numpy().tolist())
            data_true_sparse.extend(x_data.cpu().detach().numpy().tolist())
            
    return np.array(data_true),np.array(data_pred),np.array(data_recon),np.array(data_true_sparse)



seed = 666
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device_ids = [3]
device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128

## LOAD ORI DATA
hrtf_sparse = sio.loadmat('./data/hrtf_all_measured.mat')['hrtf_sparse']
hrtf_dense = sio.loadmat('./data/hrtf_all_measured.mat')['hrtf_dense']
print(hrtf_sparse.shape) # (96, 2, 440, 103)
print(hrtf_dense.shape) 
hrtf_mean = np.mean(hrtf_dense)
hrtf_std = np.std(hrtf_dense)
print(hrtf_mean)
print(hrtf_std)
hrtf_dense = (hrtf_dense - hrtf_mean)/hrtf_std
hrtf_sparse = (hrtf_sparse - hrtf_mean)/hrtf_std

K = 9
subject_all = np.arange(96)
best_model_ind_kfolds = np.load("best_model_ind_kfolds.npy")
for fold in range(K):
    print(fold)
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
    train_dataloader = DataLoader(dataset = train_dataset,batch_size = BATCH_SIZE,shuffle = False)
    valid_dataloader = DataLoader(dataset = valid_dataset,batch_size = BATCH_SIZE,shuffle = False)

    ## Initial NET
    net = hrtf_net()
    if len(device_ids) > 1:
        net = nn.DataParallel(net,device_ids=device_ids)
    net = net.to(device)
    best_model_pth = 'model/'+str(fold)+'/net'+str(int(best_model_ind_kfolds[fold,0]))+'.pth'
    state_dict = torch.load(best_model_pth)
    net.load_state_dict(state_dict)
    net.eval()

    # evalute train set
    hrtf_train_true, hrtf_train_pred, hrtf_train_recon, hrtf_train_sparse_in = test(train_dataloader)
    hrtf_train_true = hrtf_train_true*hrtf_std + hrtf_mean
    hrtf_train_pred = hrtf_train_pred*hrtf_std + hrtf_mean
    hrtf_train_recon = hrtf_train_recon*hrtf_std + hrtf_mean
    hrtf_train_sparse_in = hrtf_train_sparse_in*hrtf_std + hrtf_mean
    print(hrtf_train_true.shape)
    print(hrtf_train_pred.shape)
    print(hrtf_train_recon.shape)
    print(hrtf_train_sparse_in.shape)

    # evalute valid set
    hrtf_valid_true, hrtf_valid_pred, hrtf_valid_recon, hrtf_valid_sparse_in = test(valid_dataloader)
    hrtf_valid_true = hrtf_valid_true*hrtf_std + hrtf_mean
    hrtf_valid_pred = hrtf_valid_pred*hrtf_std + hrtf_mean
    hrtf_valid_recon = hrtf_valid_recon*hrtf_std + hrtf_mean
    hrtf_valid_sparse_in = hrtf_valid_sparse_in*hrtf_std + hrtf_mean
    print(hrtf_valid_true.shape)
    print(hrtf_valid_pred.shape)
    print(hrtf_valid_recon.shape)
    print(hrtf_valid_sparse_in.shape)


    sio.savemat('valid_data_'+str(fold)+'.mat', {'hrtf_valid_true':hrtf_valid_true,
                                                 'hrtf_valid_pred':hrtf_valid_pred,
                                                 'hrtf_valid_recon':hrtf_valid_recon,
                                                 'hrtf_valid_sparse_in':hrtf_valid_sparse_in})



