# Contains paths, constants, and device settings

import os
import torch

DATA_BASE_FOLDER = 'data/EPACC/'
TRAIN_SPLIT_PATH = os.path.join(DATA_BASE_FOLDER, 'time_series_splits/train/')
TEST_SPLIT_PATH = os.path.join(DATA_BASE_FOLDER, 'time_series_splits/test/')
FOLD_PIGS_FILE = os.path.join(DATA_BASE_FOLDER, 'fold_metadata/fold_pigs_SV.csv')
LOGGING_PROGRESS_PATH = 'Results/EPACC/Regressor/'
MODEL_PATH = 'Models/EPACC/Regressor_bl_mae/'
PLOTS_PATH = os.path.join(LOGGING_PROGRESS_PATH, 'plots/pretrain/abp_cvp_3sol/1DCNN_3_layer_SE_BN_reg/')

os.makedirs(PLOTS_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRETRAIN = True
PRINT_INTERVAL = 25
NUM_FILES = 29
NUM_EPOCHS = 150
SEQUENCE_LENGTH = 224
LEARNING_RATE = 0.0005
ADAPTIVE_LR = False
BATCH_SIZE = 16
SELECTED_MODEL = '1DCNN_3_layer_SE_BN_reg'

MODEL_NAME = f'EPACC_{SELECTED_MODEL}_model.pth'
