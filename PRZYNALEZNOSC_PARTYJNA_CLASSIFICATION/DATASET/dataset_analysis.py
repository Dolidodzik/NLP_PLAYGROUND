import pandas as pd

full_dataset = pd.read_csv('TRAINING_DATASET_RAW_2024_TO_MAY_2025.csv')
print(full_dataset)
print(full_dataset.shape)

counts = full_dataset['club'].value_counts()
print(counts)

full_dataset['word_count'] = full_dataset['statement'].str.split().map(len)
avg_wc   = full_dataset['word_count'].mean()
median_wc = full_dataset['word_count'].median()
print(f"Average word count: {avg_wc}")
print(f"Median  word count: {median_wc}")

'''
// how many statements each club has in total
club
PiS              6295
KO               5118
Konfederacja     1439
Polska2050-TD    1391
PSL-TD           1006
Lewica            980
Razem             334
Name: count, dtype: int64

Average word count: 307.0576123539893
Median word count: 189.0
'''