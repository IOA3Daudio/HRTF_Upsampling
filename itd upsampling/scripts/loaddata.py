import scipy.io as sio
import numpy as np
from scripts.datasets import myDataset
from torch.utils.data import DataLoader



class load_origin_data(object):
    """
	This is a class for loadding dataset
    Attributes:
        itd_sparse:  sparse sampling itds
        itd_sparse:  dense sampling itds
	"""
    def __init__(self,config):
        np.random.seed(config.seed)

        self.itd_sparse = sio.loadmat(config.dataset_mat_pth)['itd_sparse']
        self.itd_dense = sio.loadmat(config.dataset_mat_pth)['itd_dense']
        
        print(self.itd_sparse.shape)
        print(self.itd_dense.shape)

        self.itd_mean, self.itd_std = self.normalize_itd()

        self.BATCH_SIZE = config.batch_size
        
        
    def normalize_itd(self):
        itd_mean = np.mean(self.itd_dense)
        itd_std = np.std(self.itd_dense)
        print(itd_mean)
        print(itd_std)
        self.itd_dense = (self.itd_dense - itd_mean)/itd_std
        self.itd_sparse = (self.itd_sparse - itd_mean)/itd_std

        return itd_mean,itd_std
    
    def gen_dataloader(self,config,fold,train_shuffle_flag):
        subject_all = np.arange(config.subject_num)
        valid_subject_ind = subject_all[fold*config.valid_num:(fold+1)*config.valid_num]
        train_subject_ind = np.setdiff1d(subject_all, valid_subject_ind)
        itd_train_sparse = self.itd_sparse[train_subject_ind,:,:]
        itd_valid_sparse = self.itd_sparse[valid_subject_ind,:,:]
        itd_train_dense = self.itd_dense[train_subject_ind,:,:]
        itd_valid_dense = self.itd_dense[valid_subject_ind,:,:]

        print('Load Train Data:')
        train_dataset = myDataset(itd_train_sparse,itd_train_dense)
        print('Load Valid Data:')
        valid_dataset = myDataset(itd_valid_sparse,itd_valid_dense)
        train_dataloader = DataLoader(dataset = train_dataset,batch_size = self.BATCH_SIZE,shuffle = train_shuffle_flag)
        valid_dataloader = DataLoader(dataset = valid_dataset,batch_size = self.BATCH_SIZE,shuffle = False)

        return train_dataloader,valid_dataloader
    
        