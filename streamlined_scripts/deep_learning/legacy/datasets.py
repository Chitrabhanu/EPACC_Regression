# Contains data loading and fold splitting

import torch
import pandas as pd
import ast

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def __len__(self):
        return len(self.data)

def generate_folds(file_num, train_df, fold_pigs_file, lower_bound, upper_bound):
    idxs_to_exclude = train_df[(train_df['label'] <= lower_bound) | (train_df['label'] >= upper_bound)].index

    pigs_df = pd.read_csv(fold_pigs_file)
    row = pigs_df[pigs_df['split'] == file_num + 1].iloc[0]
    folds = [ast.literal_eval(row[f'fold_{i}']) for i in range(1, 6)]

    def extract_idxs(pigs):
        return [i for ds, pid in pigs for i in train_df.index[(train_df['dataset'] == ds) & (train_df['pig_id'] == pid)]]

    k_folds = []
    for i in range(5):
        val_pigs = folds[i]
        train_pigs = sum([folds[j] for j in range(5) if j != i], [])

        val_idx = [i for i in extract_idxs(val_pigs) if i not in idxs_to_exclude]
        train_idx = [i for i in extract_idxs(train_pigs) if i not in idxs_to_exclude]

        k_folds.append([train_idx, val_idx])

    return k_folds

def waveform_column_setup(seq_len):
    return [f'bit_{i+1}' for i in range(seq_len)]