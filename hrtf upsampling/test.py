from importlib import reload
import torch
import torch.nn as nn
import shutil
import os
import scipy.io as sio

from scripts.loaddata import load_origin_data
from scripts.models import hrtf_net
from scripts.evaluator import evaluator

import config
reload(config)
from config import DefaultConfig

def main():

    configs = DefaultConfig()

    ori_dataset = load_origin_data(configs)

    device = torch.device("cuda:{}".format(configs.device_ids[0]) if torch.cuda.is_available() else "cpu")

    for fold in range(configs.K):
        _,valid_dataloader = ori_dataset.gen_dataloader(configs,fold,train_shuffle_flag = False)

        ## Initial NET
        net = hrtf_net(configs.sparse_num)
        if len(configs.device_ids) > 1:
            net = nn.DataParallel(net,device_ids=configs.device_ids)
        net = net.to(device)
        total = sum([param.nelement() for param in net.parameters()])
        print("Number of parameter: %.2fM" % (total/1e6))
        best_model_pth = configs.model_save_pth+str(fold)+'/best_model.pt'
        state_dict = torch.load(best_model_pth)
        net.load_state_dict(state_dict)
        net.eval()

        et = evaluator(device,valid_dataloader)
        hrtf_valid_true, hrtf_valid_pred, hrtf_valid_recon, hrtf_valid_sparse_in = et.test(net)
        hrtf_valid_true = hrtf_valid_true*ori_dataset.hrtf_std + ori_dataset.hrtf_mean
        hrtf_valid_pred = hrtf_valid_pred*ori_dataset.hrtf_std + ori_dataset.hrtf_mean
        hrtf_valid_recon = hrtf_valid_recon*ori_dataset.hrtf_std + ori_dataset.hrtf_mean
        hrtf_valid_sparse_in = hrtf_valid_sparse_in*ori_dataset.hrtf_std + ori_dataset.hrtf_mean
        print(hrtf_valid_true.shape)
        print(hrtf_valid_pred.shape)
        print(hrtf_valid_recon.shape)
        print(hrtf_valid_sparse_in.shape)

        if not os.path.exists(configs.out_pth):
            os.makedirs(configs.out_pth)

        sio.savemat(configs.out_pth+'valid_data_'+str(fold)+'.mat', {'hrtf_valid_true':hrtf_valid_true,
                                                 'hrtf_valid_pred':hrtf_valid_pred,
                                                 'hrtf_valid_recon':hrtf_valid_recon,
                                                 'hrtf_valid_sparse_in':hrtf_valid_sparse_in})


if __name__ == "__main__":
    main()
