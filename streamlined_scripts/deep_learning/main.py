# Drives the full training pipeline

import pandas as pd
import torch
import numpy as np
from datasets import generate_folds, waveform_column_setup, CustomDataset
from models import *
from train import train_one_fold
from utils import weights_init
from config import *

# Load full cohort data to compute label bounds for filtering outliers
full = pd.read_csv('data/EPACC/featurized_vent_split_sample/full_vent_cohort.csv')
sv_list = full['label'].tolist()
mean, std_dev = np.mean(sv_list), np.std(sv_list)
threshold = 3 * std_dev
lower_bound, upper_bound = mean - threshold, mean + threshold

# Generate list of waveform feature column names
waveform_column_list = waveform_column_setup(SEQUENCE_LENGTH)
# Dictionary mapping model names to their corresponding class constructors
model_catalog = {
    '1DCNN_basic_3_layer': CNN1D3L,
    '1DCNN_3_layer_SE': CNN1D3LWithSE,
    '1DCNN_3_layer_SE_BN': CNN1D3LWithSEBN,
    '1DCNN_3_layer_SE_BN_reg': CNN1D3LWithSEBN_REG,
    '1DCNN_3_layer_DO': CNN1D3LWithDO,
    '1DCNN_7_layer_DO': CNN1D7LWithDO
}

# Iterate through each data split for cross-validation
for file_num in range(NUM_FILES):
    print(f"Processing split: {file_num+1}")
        # Load training data corresponding to the current split
    df = pd.read_csv(f"{TRAIN_SPLIT_PATH}SV_PS_train_{file_num+1}.csv")
        # Generate fold indices while excluding outlier labels
    folds = generate_folds(file_num, df, FOLD_PIGS_FILE, lower_bound, upper_bound)

        # Iterate over each fold within the current split
    for fold_num, (train_idx, val_idx) in enumerate(folds):
        print(f"Fold {fold_num+1}")
        # Extract features and labels
        X = df[waveform_column_list].values
        y = df['label'].values

        X_train = torch.FloatTensor([[seq] for seq in X[train_idx]])
        y_train = torch.FloatTensor(y[train_idx])
        X_val = torch.FloatTensor([[seq] for seq in X[val_idx]])
        y_val = torch.FloatTensor(y[val_idx])

         
        train_ds = CustomDataset(X_train, y_train)
        val_ds = CustomDataset(X_val, y_val)

        # Create model instance using selected architecture
        model = model_catalog[SELECTED_MODEL]()
        model.apply(weights_init)

        # Perform training and validation for this fold
        train_one_fold(model, train_ds, val_ds, fold_num + 1, file_num + 1)