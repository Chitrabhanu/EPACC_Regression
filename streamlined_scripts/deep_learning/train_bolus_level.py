#tracking model performance aggregated at the bolus level instead of at the wavelet level only
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import os
from tabulate import tabulate
from models import *
from datasets import CustomDataset
from utils import weights_init, get_adaptive_learning_rate
from config import *

def train_one_fold(model, train_dataset, val_dataset, fold_num, file_num, resume_from_epoch=0):
    import pandas as pd
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss(reduction='sum')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    train_losses, val_losses = [], []
    train_maes, val_maes = [], []
    epoch_metrics = []

    train_metadata = train_dataset.metadata.copy()
    val_metadata = val_dataset.metadata.copy()

    # Load checkpoint if resuming
    checkpoint_path = os.path.join(PLOTS_PATH, f'checkpoint_split_{file_num}_fold_{fold_num}.pt')
    if resume_from_epoch > 0 and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Resumed training from epoch {resume_from_epoch} for split {file_num}, fold {fold_num}")

    for epoch in range(resume_from_epoch, NUM_EPOCHS):
        model.train()
        total_loss, preds, targets = 0, [], []
        train_outputs = []

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(x).squeeze()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds += output.tolist()
            targets += y.tolist()
            train_outputs.append(output.detach().cpu())

        train_losses.append(total_loss)
        train_mae = mean_absolute_error(targets, preds)
        train_maes.append(train_mae)

        train_outputs = torch.cat(train_outputs, dim=0).numpy().flatten()
        train_metadata['predictions'] = train_outputs
        grouped_train = train_metadata.groupby(['pig', 'batch']).agg({
            'label': 'mean',
            'predictions': 'mean'
        }).reset_index()
        train_summary_mae = (grouped_train['label'] - grouped_train['predictions']).abs().mean()

        model.eval()
        val_loss, val_preds, val_targets = 0, [], []
        val_outputs = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                output = model(x).squeeze()
                val_loss += criterion(output, y).item()
                val_preds += output.tolist()
                val_targets += y.tolist()
                val_outputs.append(output.detach().cpu())

        val_losses.append(val_loss)
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_maes.append(val_mae)

        val_outputs = torch.cat(val_outputs, dim=0).numpy().flatten()
        val_metadata['predictions'] = val_outputs
        grouped_val = val_metadata.groupby(['pig', 'batch']).agg({
            'label': 'mean',
            'predictions': 'mean'
        }).reset_index()
        val_summary_mae = (grouped_val['label'] - grouped_val['predictions']).abs().mean()

        metrics_df = pd.DataFrame([[file_num, fold_num, epoch+1, train_mae, train_summary_mae, val_mae, val_summary_mae]],
                                  columns=['Split', 'Fold', 'Epoch', 'Train_MAE', 'Train_Bolus_MAE', 'Val_MAE', 'Val_Bolus_MAE'])
        epoch_metrics.append(metrics_df)

        if epoch % PRINT_INTERVAL == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            print(tabulate(metrics_df, headers='keys', tablefmt='psql', showindex=False))

        # Save model checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

    all_metrics_df = pd.concat(epoch_metrics, ignore_index=True)
    all_metrics_df.to_csv(os.path.join(PLOTS_PATH, f'metrics_split_{file_num}_fold_{fold_num}.csv'), index=False)

    # Plot
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig(os.path.join(PLOTS_PATH, f'loss_split_{file_num}_fold_{fold_num}.jpg'))
    plt.close()

    plt.plot(train_maes, label='Train MAE')
    plt.plot(val_maes, label='Val MAE')
    plt.legend()
    plt.title('MAE Curve')
    plt.savefig(os.path.join(PLOTS_PATH, f'mae_split_{file_num}_fold_{fold_num}.jpg'))
    plt.close()

    mae_df = pd.DataFrame({
        'Epoch': list(range(1, NUM_EPOCHS + 1)),
        'Train_MAE': train_maes,
        'Val_MAE': val_maes
    })
    mae_path = os.path.join(PLOTS_PATH, f'mae_values_split_{file_num}_fold_{fold_num}.csv')
    mae_df.to_csv(mae_path, index=False)

