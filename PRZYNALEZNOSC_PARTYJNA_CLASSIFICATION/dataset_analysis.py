import pandas as pd

df1 = pd.read_csv('2025_jan_to_may.csv')
df2 = pd.read_csv('statements_data.csv')

combined_df = pd.concat([df1, df2])

combined_df.to_csv('FULL_DATASET.csv', index=False, encoding='utf-8')