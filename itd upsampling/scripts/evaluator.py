import torch
import torch.nn as nn
import numpy as np

class evaluator(object):
    """
	This is a class for predicting HRTFs
	"""
    def __init__(self,device,dataloader):
        self.device = device
        self.dataloader = dataloader

    def test(self,net):       
        net.eval()
        data_true = []
        data_true_sparse = []
        data_pred = []
        data_recon = []
        with torch.no_grad():
            for (x_data,y_data) in self.dataloader:
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)
                out_x,out = net(x_data)

                data_true.extend(y_data.cpu().detach().numpy().tolist())
                data_pred.extend(out.cpu().detach().numpy().tolist())
                data_recon.extend(out_x.cpu().detach().numpy().tolist())
                data_true_sparse.extend(x_data.cpu().detach().numpy().tolist())
                
        return np.array(data_true),np.array(data_pred),np.array(data_recon),np.array(data_true_sparse)

