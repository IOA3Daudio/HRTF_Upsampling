import torch
import numpy as np
from torch.utils.data import Dataset

class myDataset(Dataset):
    """
	This is a class for Dataset
    Attributes:
		sparse_hrtf: sparse sampling HRTF magnitude
		target_hrtf: dense sampling HRTF magnitude
	"""
    def __init__(self, sparse_hrtf,target_hrtf):
        sparse_hrtf = torch.from_numpy(sparse_hrtf).type(torch.float32)
        target_hrtf = torch.from_numpy(target_hrtf).type(torch.float32)
        
        self.hrtfin = sparse_hrtf
        self.hrtfout = target_hrtf
    
    def __getitem__(self, index):
        return self.hrtfin[index,:,:,:], self.hrtfout[index,:,:,:]

    def __len__(self):
        return self.hrtfin.shape[0]