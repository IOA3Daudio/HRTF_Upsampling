# Head-Related Transfer Function Upsampling With Spatial Extrapolation Features
Zhao J, Yao D, Li J. Head-Related Transfer Function Upsampling With Spatial Extrapolation Features[J]. IEEE Transactions on Audio, Speech and Language Processing, 2025.

## Preprocessing
The floder named `preprocessing` houses the MATLAB code used for preprocessing the HRTFs and sampling grids.
1. Put the `*.sofa` files from the dataset (e.g. HUTUBS) into one folder, and run `main_preprocessing.m`.
2. The processed HRTF data and ITD data will be saved in corresponding `*.mat` files.


## Spectra upsampling
The floder named `hrtf upsampling` houses the Python code used for upsampling the HRTF spectra.
1. Copy the processed HRTF data into the folder `hrtf upsampling` as the path configured in `config.py`.
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


## TOA upsampling
The floder named `itd upsampling` houses the Python code used for upsampling the ITDs.
1. Copy the processed ITD data into the folder `itd upsampling` as the path configured in `config.py`.
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
