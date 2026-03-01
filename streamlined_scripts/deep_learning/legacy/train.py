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

def train_one_fold(model, train_dataset, val_dataset, fold_num, file_num):
    import pandas as pd
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss(reduction='sum')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    train_losses, val_losses = [], []
    train_maes, val_maes = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, preds, targets = 0, [], []
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

        train_losses.append(total_loss)
        train_maes.append(mean_absolute_error(targets, preds))

        model.eval()
        val_loss, val_preds, val_targets = 0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                output = model(x).squeeze()
                val_loss += criterion(output, y).item()
                val_preds += output.tolist()
                val_targets += y.tolist()

        val_losses.append(val_loss)
        val_maes.append(mean_absolute_error(val_targets, val_preds))

        if epoch % PRINT_INTERVAL == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            print(tabulate([[train_maes[-1], val_maes[-1]]], headers=["Train MAE", "Val MAE"], tablefmt='psql'))

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

    # Save MAE values to CSV
    mae_df = pd.DataFrame({
        'Epoch': list(range(1, NUM_EPOCHS + 1)),
        'Train_MAE': train_maes,
        'Val_MAE': val_maes
    })
    mae_path = os.path.join(PLOTS_PATH, f'mae_values_split_{file_num}_fold_{fold_num}.csv')
    mae_df.to_csv(mae_path, index=False)

    plt.plot(train_maes, label='Train MAE')
    plt.plot(val_maes, label='Val MAE')
    plt.legend()
    plt.title('MAE Curve')
    plt.savefig(os.path.join(PLOTS_PATH, f'mae_split_{file_num}_fold_{fold_num}.jpg'))
    plt.close()
