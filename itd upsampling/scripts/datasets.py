import torch
import numpy as np
from torch.utils.data import Dataset

class myDataset(Dataset):
    """
	This is a class for Dataset
    Attributes:
		sparse_itd: sparse sampling itds
		target_itd: dense sampling itds
	"""
    def __init__(self, sparse_itd,target_itd):
        sparse_itd = torch.from_numpy(sparse_itd).type(torch.float32)
        target_itd = torch.from_numpy(target_itd).type(torch.float32)
        
        self.itdin = sparse_itd
        self.itdout = target_itd
    
    def __getitem__(self, index):
        return self.itdin[index,:,:], self.itdout[index,:,:]

    def __len__(self):
        return self.itdin.shape[0]
