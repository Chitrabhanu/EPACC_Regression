### main.py
# Main script for evaluating a 1D CNN model on holdout datasets

from data_loading import load_filtered_data, preprocess_labels, waveform_column_list
from models import model_catalog
from utils import CustomDataset, evaluate_model, save_results
import torch
from torch.utils.data import DataLoader
import pandas as pd

# Model and data configuration
selected_model = '1DCNN_3_layer_SE_BN_reg'
data_base_folder = 'data/EPACC/'
test_split_path = data_base_folder + 'time_series_splits/test/'
model_path = 'Models/EPACC/Regressor/'
model = model_catalog[selected_model]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize result storage
rows, pred_list, tv_list, pig_list, dataset_list = [], [], [], [], []

# Iterate over holdout split files
for file_num in range(26):
    print(f"split: {file_num+1}")

    # Load and filter the test data
    test_df, pig, batch, labels, datasets = load_filtered_data(file_num+1, data_base_folder, test_split_path)
    test_df = preprocess_labels(test_df)

    # Prepare PyTorch tensors and DataLoader
    test_Xtensor = torch.FloatTensor([[seq] for seq in test_df[waveform_column_list].values.tolist()])
    test_Ytensor = torch.FloatTensor(test_df['label'].tolist())
    test_dataset = CustomDataset(test_Xtensor, test_Ytensor)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Evaluate model for the current split
    for fold in range(1):
        print(f"fold: {fold+1}")
        model_name = f"abp_cvp_pt_regressor_e50_s{file_num+1}_full.pth"
        model.load_state_dict(torch.load(model_path + model_name))
        model.eval()

        # Run evaluation and collect predictions
        pred_df, summarized_mae, overall_mae = evaluate_model(
            model, test_dataloader, pig, batch, labels, datasets
        )

        # Store predictions and evaluation metrics
        pred_list += pred_df['predictions'].tolist()
        tv_list += pred_df['true_values'].tolist()
        pig_list += pred_df['pig'].tolist()
        dataset_list += pred_df['dataset'].tolist()
        rows.append([file_num+1, fold+1, overall_mae, summarized_mae])

    print(f"completed split: {file_num + 1}")

# Save prediction outputs and evaluation summary
save_results(rows, pred_list, tv_list, pig_list, dataset_list)

# Print results as DataFrame
print(pd.DataFrame(rows, columns=['File', 'Fold', 'Wavelet_MAE', 'Bolus_MAE']))
