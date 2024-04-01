# Head-Related Transfer Function Upsampling Based on Extrapolation Features Extracted from Spatially Sparse Measurements
Jiale Zhao, Dingding Yao, and Junfeng Li: Head-Related Transfer Function Upsampling Based on Extrapolation Features Extracted from Spatially Sparse Measurements. In: Proc. IEEE/ACM Transactions on Audio Speech and Language Processing (submitted).

## Preprocessing
The floder named `preprocessing` houses the MATLAB code used for preprocessing the HRTFs and sampling grids.
1. Put the `*.sofa` files from the dataset (e.g. HUTUBS) into one folder, and run `main_preprocessing.m`.
2. The processed HRTF data and ITD data will be saved in corresponding `*.mat` files.


## Spectra upsampling
The floder named `spectra_upsampling_net` houses the Python code used for upsampling the HRTF spectra.
1. Copy the processed HRTF data into the folder `spectra_upsampling_net`.
2. run:
   ```
   python train.py
   ```
   to train the model for upsampling HRTF spectra.
4. run:
   ```
   python test.py
   ```
   to test the trained model and save the results in `*.mat`


## ITDs upsampling
The floder named `itd_upsampling_net` houses the Python code used for upsampling the ITDs.
1. Copy the processed ITD data into the folder `itd_upsampling_net`.
2. run:
   ```
   python train.py
   ```
   to train the model for upsampling ITDs.
4. run:
   ```
   python test.py
   ```
   to test the trained model and save the results in `*.mat`
