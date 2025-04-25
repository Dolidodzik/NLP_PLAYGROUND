'''
splits dataset into train, set and validation dataset in 60-20-20 proportion
'''
import pandas as pd
import numpy as np


dataset_filename = "REGRESSION_OF_FUNCTION_a=1_b=-2_c=-8_d=10.csv"
dataset_df = pd.read_csv(dataset_filename)

dataset_df = dataset_df.sample(frac=1, random_state=1111).reset_index(drop=True)

dataset_length = len(dataset_df)
print(f"loaded dataset with {dataset_length} rows")

train_size = int(0.6 * len(dataset_df))
val_size = int(0.2 * len(dataset_df)) 
test_size = len(dataset_df) - train_size - val_size

print(f"train_size: {train_size}")
print(f"val_size: {val_size}")
print(f"test_size: {test_size}")

train_df = dataset_df[:train_size]
val_df = dataset_df[train_size:train_size + val_size]
test_df = dataset_df[train_size + val_size:]

train_df.to_csv("TRAIN_" + dataset_filename, index=False)
val_df.to_csv("VALIDATION_" + dataset_filename, index=False)
test_df.to_csv("TEST_" + dataset_filename, index=False)