import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim

import os
#from google.colab import drive

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from tabulate import tabulate
import ast
from itertools import product
import math
import joblib
import pickle

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix, auc, mean_absolute_error

full = pd.read_csv('data/EPACC/featurized_vent_split_sample/full_vent_cohort.csv')

sv_list = full['label'].tolist()
mean = np.mean(sv_list)
std_dev = np.std(sv_list)

threshold = 3 * std_dev
    
lower_bound = mean - threshold
upper_bound = mean + threshold

def weights_init(m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight.data)

############################################ BASIC 1D CNN WITH 3 LAYERS ############################################


class CNN1D3L(nn.Module):
    

    
    def __init__(self):
        super(CNN1D3L, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()

        # Convolutional layer 2
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()

        # Convolutional layer 3
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()

        # Fully connected layer
        #self.fc = nn.Linear(128 * 224, 10)  # Output size: 10
        self.fc = nn.Linear(128 * 224, 2)
        #self.fc = nn.Linear(128 * 187, 2)  # Output size: 2

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc(x)

        return x






############################################ BASIC 1D CNN WITH 3 LAYERS AND SQUEEZE AND EXCITE BLOCK ############################################


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)

    def forward(self, x):
        bs, channels, _ = x.size()
        squeeze = self.avg_pool(x).view(bs, channels)
        excite = F.relu(self.fc1(squeeze))
        excite = torch.sigmoid(self.fc2(excite)).view(bs, channels, 1)
        return x * excite

class CNN1D3LWithSE(nn.Module):

    
    def __init__(self, input_channels=1, num_classes=2, reduction_ratio=16):
        super(CNN1D3LWithSE, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.se1 = SEBlock(32, reduction_ratio)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.se2 = SEBlock(64, reduction_ratio)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        #self.se3 = SEBlock(128, reduction_ratio)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.se1(self.conv1(x)))
        x = F.relu(self.se2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




    

    
############################################ BASIC 1D CNN WITH 3 LAYERS AND SQUEEZE AND EXCITE BLOCK ############################################


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)

    def forward(self, x):
        bs, channels, _ = x.size()
        squeeze = self.avg_pool(x).view(bs, channels)
        excite = F.relu(self.fc1(squeeze))
        excite = torch.sigmoid(self.fc2(excite)).view(bs, channels, 1)
        return x * excite

class CNN1D3LWithSEBN(nn.Module):

    
    
    def __init__(self, input_channels=1, num_classes=2, reduction_ratio=16):
        super(CNN1D3LWithSEBN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(32)
        #self.relu = nn.ReLU(inplace=True)
        self.se1 = SEBlock(32, reduction_ratio)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.se2 = SEBlock(64, reduction_ratio)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm1d(128)
        #self.se3 = SEBlock(128, reduction_ratio)
        
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.batchnorm1(x))
        x = self.se1(x)
        
        x = self.conv2(x)
        x = F.relu(self.batchnorm2(x))
        x = self.se2(x)
        
        x = self.conv3(x)
        x = F.relu(self.batchnorm3(x))
        #x = self.se3(x)
        
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x    
    
    


############################################ 1D CNN WITH 7 LAYERS AND DROPOUTS ############################################



class CNN1D7LWithDO(nn.Module):
    

    def __init__(self, input_channels=1, num_classes=2, dropout_rate=0.25):
        super(CNN1D7LWithDO, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(dropout_rate)
        self.conv6 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.batchnorm6 = nn.BatchNorm1d(128)
        self.dropout6 = nn.Dropout(dropout_rate)
        self.conv7 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.batchnorm7 = nn.BatchNorm1d(64)
        self.dropout7 = nn.Dropout(dropout_rate)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.dropout1(x)
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = self.dropout2(x)
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = self.dropout3(x)
        x = F.relu(self.batchnorm4(self.conv4(x)))
        x = self.dropout4(x)
        x = F.relu(self.batchnorm5(self.conv5(x)))
        x = self.dropout5(x)
        x = F.relu(self.batchnorm6(self.conv6(x)))
        x = self.dropout6(x)
        x = F.relu(self.batchnorm7(self.conv7(x)))
        x = self.dropout7(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



############################################ BASIC 1D CNN WITH 3 LAYERS AND DROPOUTS ############################################

class CNN1D3LWithDO(nn.Module):
    
    
    def __init__(self, input_channels=1, num_classes=2, dropout_rate=0.25):
        super(CNN1D3LWithDO, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.dropout1(x)
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = self.dropout2(x)
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = self.dropout3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


    
    
############################################ 1D CNN Regressor WITH 3 LAYERS AND SQUEEZE AND EXCITE BLOCK ############################################


class CNN1D3LWithSEBN_REG(nn.Module):

    
    
    def __init__(self, input_channels=1, num_classes=2, reduction_ratio=16):
        super(CNN1D3LWithSEBN_REG, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(32)
        #self.relu = nn.ReLU(inplace=True)
        self.se1 = SEBlock(32, reduction_ratio)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.se2 = SEBlock(64, reduction_ratio)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm1d(128)
        #self.se3 = SEBlock(128, reduction_ratio)
        
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 1)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.batchnorm1(x))
        x = self.se1(x)
        
        x = self.conv2(x)
        x = F.relu(self.batchnorm2(x))
        x = self.se2(x)
        
        x = self.conv3(x)
        x = F.relu(self.batchnorm3(x))
        #x = self.se3(x)
        
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 
    
    



############################################ OTHER PROCESSING AND HELPER FUNCTIONS ############################################



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        return x, y

    def __len__(self):
        return len(self.data)


def generate_folds(file_num, train_df, fold_pigs_file):
    
    indexes_to_remove = train_df[(train_df['label'] <= lower_bound) | 
                                        (train_df['label'] >= upper_bound)].index.tolist()


    fold_pigs_df = pd.read_csv(fold_pigs_file)
    folds_for_current_split = fold_pigs_df[fold_pigs_df['split']==file_num+1]
    fold_1_pigs = ast.literal_eval(folds_for_current_split['fold_1'].tolist()[0])
    fold_2_pigs = ast.literal_eval(folds_for_current_split['fold_2'].tolist()[0])
    fold_3_pigs = ast.literal_eval(folds_for_current_split['fold_3'].tolist()[0])
    fold_4_pigs = ast.literal_eval(folds_for_current_split['fold_4'].tolist()[0])
    fold_5_pigs = ast.literal_eval(folds_for_current_split['fold_5'].tolist()[0])
    all_fold_pigs = [fold_1_pigs,fold_2_pigs,fold_3_pigs,fold_4_pigs,fold_5_pigs]


    pig_id_col = 'pig_id'
    k_folds = []
    for fold in all_fold_pigs:

        train_pig_list = [f for f in all_fold_pigs if f!=fold]
        train_pigs = []
        for l in range(len(train_pig_list)):
            train_pigs+=train_pig_list[l]

        val_idxs = []
        train_idxs = []
        for p in range(len(fold)):
            idx = train_df.index[(train_df['dataset'] == fold[p][0]) & (train_df[pig_id_col] == fold[p][1])].tolist()
            val_idxs+=idx
        val_idxs = [i for i in val_idxs if i not in indexes_to_remove]
        #print(val_idxs)
        for p in range(len(train_pigs)):
            idx = train_df.index[(train_df['dataset'] == train_pigs[p][0]) & (train_df[pig_id_col] == train_pigs[p][1])].tolist()
            train_idxs+=idx
        train_idxs = [i for i in train_idxs if i not in indexes_to_remove]
        #print(train_idxs)
        #k_folds.append((np.array(train_idxs), np.array(val_idxs)))
        k_folds.append([train_idxs, val_idxs])

    return k_folds


def grid_generator(param_dict):
    list_of_param_lists = []
    for key in list(param_dict.keys()):
        list_of_param_lists.append(param_dict[key])

    all_combinations = list(product(*list_of_param_lists))
    all_combinations = [list(combo) for combo in all_combinations]
    return all_combinations

def get_adaptive_learning_rate(epoch, initial_rate=0.01, step_size = 50):
    power = int(epoch/step_size)
    rate = initial_rate/10**power
    return rate
    


selected_model = '1DCNN_3_layer_SE_BN_reg'


#drive.mount('/content/drive')
data_base_folder = 'data/EPACC/'

train_split_path = data_base_folder+'time_series_splits/train/'
test_split_path = data_base_folder+'time_series_splits/test/'

#train_split_path = data_base_folder+'time_series_splits_6k/train/'
#test_split_path = data_base_folder+'time_series_splits_6k/test/'

fold_pigs_folder = data_base_folder+'fold_metadata/'
fold_pigs_file = fold_pigs_folder+'fold_pigs_SV.csv'

logging_progress_path = 'Results/EPACC/Regressor/'
model_path = 'Models/EPACC/Regressor_bl_mae/'
plots_path = logging_progress_path + 'plots/pretrain/abp_cvp_3sol/'+ selected_model + '/'
if not os.path.isdir(plots_path):
    os.makedirs(plots_path)
if not os.path.isdir(model_path):
    os.makedirs(model_path)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrain = True
pretrain_loaded = False
print_interval = 25
num_files=29
num_epochs = 150
sequence_length = 224 #6000
adaptive_learning_rate = False
if not adaptive_learning_rate:
    learning_rate = 0.0005
waveform_column_list = ['bit_'+str(i+1) for i in range(sequence_length)]

last_bits = True
if last_bits == True:
    n_bits = 224
    waveform_column_list=waveform_column_list[-1*n_bits:]

start_mode = 'Fresh' #'Continue'
model_name = 'EPACC_'+selected_model+'_model.pth'
training_log_file = 'EPACC_'+selected_model+'_model_training_metrics.csv'
training_losses_file = 'EPACC_'+selected_model+'_model_training_losses.pkl'
training_accuracies_file = 'EPACC_'+selected_model+'_model_training_accuracies.pkl'
training_aurocs_file = 'EPACC_'+selected_model+'_model_training_aurocs.pkl'
validation_log_file = 'EPACC_'+selected_model+'_model_validation_metrics.csv'
validation_losses_file = 'EPACC_'+selected_model+'_model_validation_losses.pkl'
validation_accuracies_file = 'EPACC_'+selected_model+'_model_validation_accuracies.pkl'
validation_aurocs_file = 'EPACC_'+selected_model+'_model_validation_aurocs.pkl'
final_fold_performance = 'EPACC_'+selected_model+'_model_avg_fold_validation_performance.csv'


model_catalog = {
    '1DCNN_basic_3_layer': CNN1D3L(), #1dcnn with 3 layers
    '1DCNN_3_layer_SE': CNN1D3LWithSE(), #1dcnn with 3 layers and squeeze and excite blocks
    '1DCNN_3_layer_SE_BN': CNN1D3LWithSEBN(), #1dcnn with 3 layers and squeeze and excite blocks and Batchnorm
    '1DCNN_3_layer_DO': CNN1D3LWithDO,
    '1DCNN_7_layer_DO':CNN1D7LWithDO(), #1dcnn with 7 layers and dropouts
    '1DCNN_3_layer_SE_BN_reg':CNN1D3LWithSEBN_REG() 
}



if start_mode == 'Continue':
    train_log_df = pd.read_csv(logging_progress_path+training_log_file)
    last_file_num = train_log_df['File'].tolist()[0]
    print('Epoch: '+ str(train_log_df['Epoch'].tolist()[0]))
else:
    last_file_num = 1


train_pred_list = []
train_tv_list = []
val_pred_list = []
val_tv_list = []

for file_num in range(last_file_num-1, num_files):
    
    print("split: " + str(file_num+1))
    
    train_file = 'SV_PS_train_' + str(file_num+1) + '.csv'
    train_df = pd.read_csv(train_split_path+train_file)
    k_folds = generate_folds(file_num, train_df, fold_pigs_file)

    if start_mode == 'Continue':
        train_log_df = pd.read_csv(logging_progress_path+training_log_file)
        last_fold_num = train_log_df['Fold'].tolist()[0]
        k_folds = k_folds[last_fold_num-1:]

    fold_performance = []
    for fold_num, fold in enumerate(k_folds):


        print('Fold ' + str(fold_num+1))
        
        train_df_id = train_df[['pig', 'batch', 'label']]
        train_df_id = train_df_id.rename(columns={'label': 'true_values'})
        train_df_id_t_fold = train_df_id.iloc[fold[0]]
        train_df_id_v_fold = train_df_id.iloc[fold[1]]
        
        
        train_df_X = train_df[waveform_column_list]
        train_df_Y = train_df[['label']]

        train_df_X_t_fold = train_df_X.iloc[fold[0]]
        train_df_Y_t_fold = train_df_Y.iloc[fold[0]]
        train_seq_list_X_t_fold = [[seq] for seq in train_df_X_t_fold.values.tolist()]
        train_Xtensor_t_fold = torch.FloatTensor(train_seq_list_X_t_fold)
        train_Ytensor_t_fold = torch.FloatTensor(train_df_Y_t_fold['label'].tolist())

        train_df_X_v_fold = train_df_X.iloc[fold[1]]
        train_df_Y_v_fold = train_df_Y.iloc[fold[1]]
        train_seq_list_X_v_fold = [[seq] for seq in train_df_X_v_fold.values.tolist()]
        train_Xtensor_v_fold = torch.FloatTensor(train_seq_list_X_v_fold)
        train_Ytensor_v_fold = torch.FloatTensor(train_df_Y_v_fold['label'].tolist())


        train_dataset_t_fold = CustomDataset(train_Xtensor_t_fold, train_Ytensor_t_fold)
        train_dataset_v_fold = CustomDataset(train_Xtensor_v_fold, train_Ytensor_v_fold)



        #save for plots
        if start_mode == 'Continue':
            accuracies = joblib.load(model_path+training_accuracies_file)
            val_accuracies = joblib.load(model_path+validation_accuracies_file)
            aurocs = joblib.load(model_path+training_aurocs_file)
            val_aurocs = joblib.load(model_path+validation_aurocs_file)
            losses = joblib.load(model_path+training_losses_file)
            val_losses = joblib.load(model_path+validation_losses_file)
        else:
            mae = []
            val_maes = []
            aurocs = []
            val_aurocs = []
            losses = []
            val_losses = []
        # Define the batch size
        batch_size = 16

        # Create a data loader
        train_dataloader = DataLoader(train_dataset_t_fold, batch_size=batch_size, shuffle=True)

        # Create an instance of the CNN model
        if start_mode == 'Continue':
            model = model_catalog[selected_model]
            model.load_state_dict(torch.load(model_path+model_name))
        else:
            model = model_catalog[selected_model] #CNN1D3L()
            model.apply(weights_init)
            
        if pretrain:
            if not pretrain_loaded:
                print('Loading pretrained weights into model...')
                #pretrained_model = model_catalog[selected_model]

                # Load the state dictionary of the source model
                source_state_dict = torch.load('pretrainer_state_dict.pth')
                
                # Create a new state dictionary for the current model, copying only the convolutional layer weights
                new_state_dict = model.state_dict()
                for name, param in source_state_dict.items():
                    if 'conv' in name or 'batch' in name:  # Copy only conv and batch norm layers
                        new_state_dict[name] = param
                model.load_state_dict(new_state_dict)
                model.to(device)
                pretrain_loaded = True

        # Define the loss function
        criterion = nn.MSELoss(reduction='sum')

        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        if start_mode == 'Continue':
            train_log_df = pd.read_csv(logging_progress_path+training_log_file)
            last_epoch_num = train_log_df['Epoch'].tolist()[0]

        else:
            last_epoch_num = 0

        epoch_mae_train, epoch_mae_val = [], []
        for epoch in range(last_epoch_num, num_epochs):
            
            #print('Starting epoch ' + str(epoch+1))
            #print('')

            if adaptive_learning_rate:
                learning_rate = get_adaptive_learning_rate(epoch)
            epoch_loss = 0.0
            num_batches=0

            total = 0
            correct = 0
            predicted_scores = []
            true_labels = []

            train_mae = []
            pred_list = []
            
            train_ops = []
            
            for batch_data, batch_targets in train_dataloader:
                
                
                num_batches+=1
                
                
                batch_data = batch_data.to(device)
                batch_targets = batch_targets.to(device)
                # Zero the gradients
                optimizer.zero_grad()
                # Forward pass
                output = model(batch_data)
                # Calculate the loss
                output_squeezed = output.squeeze(dim=1)
                pred_list += output_squeezed
                loss = criterion(output_squeezed, batch_targets)
                # Backward pass
                loss.backward()
                # Update the weights
                optimizer.step()


                # Calculate epoch loss
                epoch_loss += loss.item()
                # Calculate accuracy
                
                #print('OP_Sq: ', output_squeezed)
                #print('OP: ', output)
                
                total += batch_targets.size(0)
                #train_mae.append(mean_absolute_error(batch_targets.detach().numpy(), output_squeezed.detach().numpy()))
                true_labels.extend(batch_targets.tolist())
                
                train_ops.append(output.detach().cpu())


            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss #/ num_batches

            #avg_train_mae = sum(train_mae) / len(train_mae)
            avg_train_mae = mean_absolute_error(torch.tensor(true_labels, dtype=torch.float32).clone().detach().numpy(), torch.tensor(pred_list, dtype=torch.float32).clone().detach().numpy())
            epoch_mae_train.append(avg_train_mae)
            
            train_ops = torch.cat(train_ops, dim=0)
            train_all_predictions_list = train_ops.tolist()
            train_df_id_t_fold['predictions'] = [l[0] for l in train_all_predictions_list]
            train_pred_df_summarized = train_df_id_t_fold.groupby(['pig', 'batch']).agg({
                                                                                  'true_values': 'mean',    
                                                                                  'predictions': 'mean'   
                                                                                  }).reset_index()
            train_pred_list+=train_pred_df_summarized['predictions'].tolist()
            train_tv_list+=train_pred_df_summarized['true_values'].tolist()
            train_summarized_mae = (train_pred_df_summarized['true_values'] - train_pred_df_summarized['predictions']).abs().mean()


            # Print the loss for each epoch
            #print('')
            #print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss}")
            #print('accuracy : ' + str(round(100*accuracy,2)) + '%')
            #print('')
            metrics_df = pd.DataFrame([[file_num+1, fold_num+1, epoch, avg_train_mae, train_summarized_mae]],
                                      columns = ['split', 'fold', 'epoch', 'wavelet_MAE', 'bolus_MAE'])
            if epoch%print_interval==0:
                print('Starting epoch ' + str(epoch+1))
                print('')
                print('Train performance:')
                print(tabulate(metrics_df, headers='keys', tablefmt='psql', showindex=False))
                print('')
                print('----------------------------------------------------------------------------------------')

            losses.append(round(avg_epoch_loss,2))
            mae.append(round(avg_train_mae, 2))
            


            #Save Progress
            if epoch%10==0:
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                torch.save(model.state_dict(), model_path+model_name)
                #joblib.dump(losses, model_path+training_losses_file)
                #joblib.dump(accuracies, model_path+training_accuracies_file)
                #joblib.dump(aurocs, model_path+training_aurocs_file)

                if not os.path.isdir(logging_progress_path):
                    os.makedirs(logging_progress_path)
                #metrics_df['Epoch'] = [epoch]
                #metrics_df['Fold'] = [fold_num+1]
                #metrics_df['File'] = [file_num+1]
                #metrics_df.to_csv(logging_progress_path+training_log_file, index=False)
                
            
            






            # Validation Loop
            model.eval()
            total = 0
            correct = 0
            num_batches=0
            epoch_val_loss = 0.0
            predicted_scores = []
            true_labels = []
            val_mae = []
            pred_list = []

            # Create a data loader
            val_dataloader = DataLoader(train_dataset_v_fold, batch_size=batch_size, shuffle=True)
            
            
            val_ops = []

            with torch.no_grad():
                for batch_data, batch_labels in val_dataloader:
                
                
                    num_batches+=1
                    
                    
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    output = model(batch_data)
                    output_squeezed = output.squeeze(dim=1)
                    pred_list += output_squeezed
                    #Calculate validation loss
                    
                    loss = criterion(output_squeezed, batch_labels)
                    epoch_val_loss += loss.item()
                    # Calculate accuracy
                    
                    total += batch_labels.size(0)
                    #val_mae.append()
                    true_labels.extend(batch_labels.tolist())
                    
                    val_ops.append(output.detach().cpu())

                # Calculate average epoch loss
                avg_epoch_val_loss = epoch_val_loss #/ num_batches
                #avg_val_mae = sum(val_mae) / len(val_mae)
                
                avg_val_mae = mean_absolute_error(torch.tensor(batch_labels, dtype=torch.float32).clone().detach().cpu().numpy(), torch.tensor(output_squeezed, dtype=torch.float32).clone().detach().cpu().numpy())
                epoch_mae_val.append(avg_val_mae)
                
                val_ops = torch.cat(val_ops, dim=0)
                val_all_predictions_list = val_ops.tolist()
                train_df_id_v_fold['predictions'] = [l[0] for l in val_all_predictions_list]
                val_pred_df_summarized = train_df_id_v_fold.groupby(['pig', 'batch']).agg({
                                                                                      'true_values': 'mean',    
                                                                                      'predictions': 'mean'   
                                                                                      }).reset_index()
                val_pred_list+=val_pred_df_summarized['predictions'].tolist()
                val_tv_list+=val_pred_df_summarized['true_values'].tolist()
                val_summarized_mae = (val_pred_df_summarized['true_values'] - val_pred_df_summarized['predictions']).abs().mean()
                

            metrics_df = pd.DataFrame([[avg_val_mae, val_summarized_mae]],
                                      columns = ['wavelet_MAE', 'bolus_MAE'])

            if epoch%print_interval==0:
                print('Validation performance:')
                print(tabulate(metrics_df, headers='keys', tablefmt='psql', showindex=False))

                print('')
                print('----------------------------------------------------------------------------------------')

            val_losses.append(round(avg_epoch_val_loss,2))
            val_maes.append(round(avg_val_mae, 2))


            #Save Progress
            #if epoch%10==0:
            #    joblib.dump(val_losses, model_path+validation_losses_file)
            #    joblib.dump(val_accuracies, model_path+validation_accuracies_file)
            #    joblib.dump(val_aurocs, model_path+validation_aurocs_file)
            #    if not os.path.isdir(logging_progress_path):
            #        os.makedirs(logging_progress_path)
            #    metrics_df['Epoch'] = [epoch]
            #    metrics_df['Fold'] = [fold_num+1]
            #    metrics_df['File'] = [file_num+1]
            #    metrics_df.to_csv(logging_progress_path+validation_log_file, index=False)
            
            
            
            if epoch==20:
                metrics_df['Epoch'] = [epoch]*metrics_df.shape[0]
                metrics_df['Fold'] = [fold_num+1]*metrics_df.shape[0]
                metrics_df['File'] = [file_num+1]*metrics_df.shape[0]
                metrics_df.to_csv('bl_pretrained_weights_3LBNSE_abp_cvp_3sol_reg_v4_epoch20.csv', index=False, mode='a')
                
            if epoch==35:
                metrics_df['Epoch'] = [epoch]*metrics_df.shape[0]
                metrics_df['Fold'] = [fold_num+1]*metrics_df.shape[0]
                metrics_df['File'] = [file_num+1]*metrics_df.shape[0]
                metrics_df.to_csv('bl_pretrained_weights_3LBNSE_abp_cvp_3sol_reg_v4_epoch35.csv', index=False, mode='a')
                
            if epoch==50:
                torch.save(model.state_dict(), model_path + 'abp_cvp_pt_regressor_e50_s' + str(file_num+1) + '_f' + str(fold_num+1) + '.pth')
                metrics_df['Epoch'] = [epoch]*metrics_df.shape[0]
                metrics_df['Fold'] = [fold_num+1]*metrics_df.shape[0]
                metrics_df['File'] = [file_num+1]*metrics_df.shape[0]
                metrics_df.to_csv('bl_pretrained_weights_3LBNSE_abp_cvp_3sol_reg_v4_epoch50.csv', index=False, mode='a')
                
            if epoch==num_epochs-1:
                torch.save(model.state_dict(), model_path + 'abp_cvp_pt_regressor_e150_s' + str(file_num+1) + '_f' + str(fold_num+1) + '.pth')
                metrics_df['Epoch'] = [epoch]*metrics_df.shape[0]
                metrics_df['Fold'] = [fold_num+1]*metrics_df.shape[0]
                metrics_df['File'] = [file_num+1]*metrics_df.shape[0]
                metrics_df.to_csv('bl_pretrained_weights_3LBNSE_abp_cvp_3sol_reg_v4_epoch150.csv', index=False, mode='a')

        
        plt.close()
        plt.plot(losses, label = 'Training Loss')
        plt.plot(val_losses, label = 'Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.show()
        plt.savefig(plots_path + 'Loss_split_' + str(file_num+1) + '_fold_' + str(fold_num+1) + '.jpg')
        plt.close()


        plt.plot(epoch_mae_train, label = 'Training MAE')
        plt.plot(epoch_mae_val, label = 'Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('MAE Curve')
        plt.legend()
        plt.show()
        plt.savefig(plots_path + 'MAE_split_' + str(file_num+1) + '_fold_' + str(fold_num+1) + '.jpg')
        plt.close()







        fold_performance.append(metrics_df)


with open('train_preds.pkl', 'wb') as file:
    pickle.dump(train_pred_list, file)
with open('train_tvs.pkl', 'wb') as file:
    pickle.dump(train_tv_list, file)
with open('val_preds.pkl', 'wb') as file:
    pickle.dump(val_pred_list, file)
with open('val_tvs.pkl', 'wb') as file:
    pickle.dump(val_tv_list, file)

    #metrics = ['MAE']
    #avg_performance_rows = []
    #metric_ci_rows = []
    #for metric in metrics:
    #    metric_vals = []
    #    for f in range(len(fold_performance)):
    #        metric_vals.append(fold_performance[f][metric].tolist()[0])
    #    avg_metric_val = round(sum(metric_vals)/len(metric_vals),2)
    #    ci = round(1.96*np.std(metric_vals)/np.sqrt(len(metric_vals)),3)
    #    avg_performance_rows.append(avg_metric_val)
    #    metric_ci_rows.append(ci)
#
    #avg_fold_performance = pd.DataFrame([avg_performance_rows+metric_ci_rows],
    #                                    columns = metrics + [m+'_95%_CI' for m in metrics])
    #avg_fold_performance.to_csv(logging_progress_path+final_fold_performance)