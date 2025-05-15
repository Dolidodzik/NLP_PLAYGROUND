import pandas as pd
from sklearn.model_selection import train_test_split

# N is just max amount of statements per club
N = 1000
dataset_df = pd.read_csv('CLEANED_DATASET_2024_TO_MAY_2025.csv')
dataset_df = dataset_df.sample(frac=1).reset_index(drop=True)
print("working on dataset:")
print(dataset_df)
print("\n ========================== \n")

unique_clubs = dataset_df['club'].unique().tolist()
print("unique clubs found: ", unique_clubs)
print("club statements count:")
print(dataset_df['club'].value_counts())

limited_dataset_df = (
    dataset_df
    .groupby('club', group_keys=False)
    .apply(lambda grp: grp.sample(n=min(len(grp), N)))
    .reset_index(drop=True)
)

print(limited_dataset_df)
print(limited_dataset_df['club'].value_counts())

train_df, validation_df = train_test_split(
    limited_dataset_df,
    test_size=0.2,
    stratify=limited_dataset_df['club']
)

train_df.to_csv('TRAINING_DATASET_RAW.csv', index=False)
validation_df.to_csv('VALIDATION_DATASET.csv', index=False)