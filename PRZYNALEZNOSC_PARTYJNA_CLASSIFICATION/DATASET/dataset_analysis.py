import pandas as pd

full_dataset = pd.read_csv('FULL_DATASET_ORIGINAL_2024_TO_MAY_2025.csv')

counts = full_dataset['club'].value_counts()
print(counts)

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
'''