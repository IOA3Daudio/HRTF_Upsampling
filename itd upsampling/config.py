
class DefaultConfig(object):
    """
	This is a class for configurations
	"""

    # GPU list
    device_ids = [0]
    # Batch size
    batch_size = 84
    # EPOCHS
    epochs = 800
    # Original Dataset Path
    dataset_mat_pth = './dataset mats/itd_all_measured.mat'
    # Output Save Path
    model_save_pth = './model/'
    out_pth = './itd out/'
    loss_png_pth = './loss/'

    #K folds cross-validation
    K = 9
    # total subjects
    subject_num = 94
    # number of subjects in valid dataset
    valid_num = 10
    # sparse resolution
    sparse_num = 121

    
    # random seed
    seed = 666

    scheduler_step_size = 200
    scheduler_gamma = 0.9
    
