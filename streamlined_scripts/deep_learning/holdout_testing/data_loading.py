# data_loading.py

import pandas as pd
import numpy as np

# Load the full dataset to calculate label thresholds
full = pd.read_csv('data/EPACC/featurized_vent_split_sample/full_vent_cohort.csv')
sv_list = full['label'].tolist()
mean = np.mean(sv_list)
std_dev = np.std(sv_list)
threshold = 3 * std_dev
lower_bound = mean - threshold
upper_bound = mean + threshold

# Column list for waveform features
sequence_length = 224
waveform_column_list = ['bit_' + str(i + 1) for i in range(sequence_length)]

# Function to load and filter test data

def load_filtered_data(file_index, base_path, split_path):
    test_file = f'SV_PS_test_{file_index}.csv'
    test_df = pd.read_csv(split_path + test_file)

    # Filter using additional metadata file
    filter_df = pd.read_csv(base_path + f'time_series_splits/test_3so_removed/combined_chort_test_{file_index}_v0.csv')
    filter_df = filter_df[['pig', 'dataset']]
    merged_df = pd.merge(filter_df, test_df, on=['pig', 'dataset'], how='inner')
    test_df = merged_df[test_df.columns]

    # Extract metadata columns
    pig = test_df['pig'].tolist()
    batch = test_df['batch'].tolist()
    labels = test_df['label'].tolist()
    datasets = test_df['dataset'].tolist()

    return test_df, pig, batch, labels, datasets

# Function to apply label bounds and remove outliers

def preprocess_labels(df):
    return df[(df['label'] > lower_bound) & (df['label'] < upper_bound)]
