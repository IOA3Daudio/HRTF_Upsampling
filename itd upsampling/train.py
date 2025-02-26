from importlib import reload
import torch
import torch.nn as nn
import shutil
import os
import scipy.io as sio

from scripts.loaddata import load_origin_data
from scripts.models import itd_net
from scripts.trainer import trainer
from scripts.utils import plot_loss
from scripts.evaluator import evaluator

import config
reload(config)
from config import DefaultConfig

def main():

    configs = DefaultConfig()

    ori_dataset = load_origin_data(configs)

    device = torch.device("cuda:{}".format(configs.device_ids[0]) if torch.cuda.is_available() else "cpu")

    lossF = nn.MSELoss()

    for fold in range(configs.K):
        train_dataloader,valid_dataloader = ori_dataset.gen_dataloader(configs,fold,train_shuffle_flag = True)

        ## Initial NET
        net = itd_net(configs.sparse_num)
        if len(configs.device_ids) > 1:
            net = nn.DataParallel(net,device_ids=configs.device_ids)
        net = net.to(device)
        total = sum([param.nelement() for param in net.parameters()])
        print("Number of parameter: %.2fM" % (total/1e6))
        
        optimizer = torch.optim.Adam(net.parameters())

        mytrainer = trainer(configs,fold,device,train_dataloader,valid_dataloader,optimizer,lossF)

        ## Train
        history = {'Train Loss':[],'Valid Loss':[]}
        torch.cuda.empty_cache()
        for epoch in range(configs.epochs):
            mytrainer.train(net,epoch,history)
            mytrainer.valid(net,history)
        
        net.train(False)

        best_model_ind = plot_loss(configs,history,fold)
        best_model_state_dict = torch.load(configs.model_save_pth+str(fold)+'/net'+str(best_model_ind)+'.pt')
        shutil.rmtree(configs.model_save_pth+str(fold))
        os.mkdir(configs.model_save_pth+str(fold))
        torch.save(best_model_state_dict,configs.model_save_pth+str(fold)+'/best_model.pt')

if __name__ == "__main__":
    main()
