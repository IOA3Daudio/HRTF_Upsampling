import os
import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data_processing import myDataset
from build_net import itd_net
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

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
device_ids = [2]
device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128

## LOAD ORI DATA
itd_sparse = sio.loadmat('./data/itd_all_measured.mat')['itd_sparse']
itd_dense = sio.loadmat('./data/itd_all_measured.mat')['itd_dense']
print(itd_sparse.shape)
print(itd_dense.shape) 
itd_mean = np.mean(itd_dense)
itd_std = np.std(itd_dense)
print(itd_mean)
print(itd_std)
itd_dense = (itd_dense - itd_mean)/itd_std
itd_sparse = (itd_sparse - itd_mean)/itd_std

K = 9
subject_all = np.arange(96)
best_model_ind_kfolds = np.load("best_model_ind_kfolds.npy")
for fold in range(K):
    print(fold)
    valid_subject_ind = subject_all[fold*10:(fold+1)*10]
    train_subject_ind = np.setdiff1d(subject_all, valid_subject_ind)
    itd_train_sparse = itd_sparse[train_subject_ind,:,:]
    itd_valid_sparse = itd_sparse[valid_subject_ind,:,:]
    itd_train_dense = itd_dense[train_subject_ind,:,:]
    itd_valid_dense = itd_dense[valid_subject_ind,:,:]

    print('Load Train Data:')
    train_dataset = myDataset(itd_train_sparse,itd_train_dense)
    print('Load Valid Data:')
    valid_dataset = myDataset(itd_valid_sparse,itd_valid_dense)
    train_dataloader = DataLoader(dataset = train_dataset,batch_size = BATCH_SIZE,shuffle = False)
    valid_dataloader = DataLoader(dataset = valid_dataset,batch_size = BATCH_SIZE,shuffle = False)

    ## Initial NET
    net = itd_net()
    if len(device_ids) > 1:
        net = nn.DataParallel(net,device_ids=device_ids)
    net = net.to(device)
    best_model_pth = 'model/'+str(fold)+'/net'+str(int(best_model_ind_kfolds[fold,0]))+'.pth'
    state_dict = torch.load(best_model_pth)
    net.load_state_dict(state_dict)
    net.eval()

    # evalute train set
    itd_train_true, itd_train_pred, itd_train_recon, itd_train_sparse_in = test(train_dataloader)
    itd_train_true = itd_train_true*itd_std + itd_mean
    itd_train_pred = itd_train_pred*itd_std + itd_mean
    itd_train_recon = itd_train_recon*itd_std + itd_mean
    itd_train_sparse_in = itd_train_sparse_in*itd_std + itd_mean
    print(itd_train_true.shape)
    print(itd_train_pred.shape)
    print(itd_train_recon.shape)
    print(itd_train_sparse_in.shape)

    # evalute valid set
    itd_valid_true, itd_valid_pred, itd_valid_recon, itd_valid_sparse_in = test(valid_dataloader)
    itd_valid_true = itd_valid_true*itd_std + itd_mean
    itd_valid_pred = itd_valid_pred*itd_std + itd_mean
    itd_valid_recon = itd_valid_recon*itd_std + itd_mean
    itd_valid_sparse_in = itd_valid_sparse_in*itd_std + itd_mean
    print(itd_valid_true.shape)
    print(itd_valid_pred.shape)
    print(itd_valid_recon.shape)
    print(itd_valid_sparse_in.shape)


    sio.savemat('valid_data_'+str(fold)+'.mat', {'itd_valid_true':itd_valid_true,
                                                 'itd_valid_pred':itd_valid_pred,
                                                 'itd_valid_recon':itd_valid_recon,
                                                 'itd_valid_sparse_in':itd_valid_sparse_in})



