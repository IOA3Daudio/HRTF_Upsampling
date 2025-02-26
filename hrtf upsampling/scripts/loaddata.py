import scipy.io as sio
import numpy as np
from scripts.datasets import myDataset
from torch.utils.data import DataLoader



class load_origin_data(object):
    """
	This is a class for loadding dataset
    Attributes:
        hrtf_sparse:  sparse sampling HRTF magnitude
        hrtf_sparse:  dense sampling HRTF magnitude
	"""
    def __init__(self,config):
        np.random.seed(config.seed)

        self.hrtf_sparse = sio.loadmat(config.dataset_mat_pth)['hrtf_sparse']
        self.hrtf_dense = sio.loadmat(config.dataset_mat_pth)['hrtf_dense']
        
        print(self.hrtf_sparse.shape)
        print(self.hrtf_dense.shape)

        self.hrtf_mean, self.hrtf_std = self.normalize_hrtf()

        self.BATCH_SIZE = config.batch_size
        
        
    def normalize_hrtf(self):
        hrtf_mean = np.mean(self.hrtf_dense)
        hrtf_std = np.std(self.hrtf_dense)
        print(hrtf_mean)
        print(hrtf_std)
        self.hrtf_dense = (self.hrtf_dense - hrtf_mean)/hrtf_std
        self.hrtf_sparse = (self.hrtf_sparse - hrtf_mean)/hrtf_std

        return hrtf_mean,hrtf_std
    
    def gen_dataloader(self,config,fold,train_shuffle_flag):
        subject_all = np.arange(config.subject_num)
        valid_subject_ind = subject_all[fold*config.valid_num:(fold+1)*config.valid_num]
        train_subject_ind = np.setdiff1d(subject_all, valid_subject_ind)
        hrtf_train_sparse = self.hrtf_sparse[train_subject_ind,:,:,:]
        hrtf_valid_sparse = self.hrtf_sparse[valid_subject_ind,:,:,:]
        hrtf_train_dense = self.hrtf_dense[train_subject_ind,:,:,:]
        hrtf_valid_dense = self.hrtf_dense[valid_subject_ind,:,:,:]

        print('Load Train Data:')
        train_dataset = myDataset(hrtf_train_sparse,hrtf_train_dense)
        print('Load Valid Data:')
        valid_dataset = myDataset(hrtf_valid_sparse,hrtf_valid_dense)
        train_dataloader = DataLoader(dataset = train_dataset,batch_size = self.BATCH_SIZE,shuffle = train_shuffle_flag)
        valid_dataloader = DataLoader(dataset = valid_dataset,batch_size = self.BATCH_SIZE,shuffle = False)

        return train_dataloader,valid_dataloader
    
        