# utils.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import pickle

# Custom PyTorch Dataset for (X, y) tensor pairs
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)

# Evaluate model and return prediction DataFrame and error metrics
def evaluate_model(model, dataloader, pigs, batches, labels, datasets):
    criterion = nn.MSELoss()
    model.eval()

    total_absolute_error = 0.0
    total_samples = 0
    outputs_collected = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_absolute_error += torch.sum(torch.abs(outputs - targets)).item()
            total_samples += inputs.size(0)
            outputs_collected.append(outputs.detach().cpu())

    all_outputs = torch.cat(outputs_collected, dim=0).tolist()

    pred_df = pd.DataFrame(pigs, columns=["pig"])
    pred_df['batch'] = batches
    pred_df['dataset'] = datasets
    pred_df['predictions'] = [o[0] for o in all_outputs]
    pred_df['true_values'] = labels

    pred_df_summary = pred_df.groupby(['pig', 'batch', 'dataset']).agg({
        'true_values': 'max',
        'predictions': 'mean'
    }).reset_index()

    summarized_mae = (pred_df_summary['true_values'] - pred_df_summary['predictions']).abs().mean()
    overall_mae = round(total_absolute_error / total_samples, 2)

    return pred_df_summary, summarized_mae, overall_mae

# Save predictions and results to files
def save_results(rows, pred_list, tv_list, pig_list, dataset_list):
    with open('holdout_preds_full_train.pkl', 'wb') as f:
        pickle.dump(pred_list, f)
    with open('holdout_tvs_full_train.pkl', 'wb') as f:
        pickle.dump(tv_list, f)
    with open('holdout_pigs_full_train.pkl', 'wb') as f:
        pickle.dump(pig_list, f)
    with open('holdout_datasets_full_train.pkl', 'wb') as f:
        pickle.dump(dataset_list, f)

    results_df = pd.DataFrame(rows, columns=['File', 'Fold', 'Wavelet_MAE', 'Bolus_MAE'])
    results_df.to_csv('holdout_results__full_train_v5.csv', index=False)
