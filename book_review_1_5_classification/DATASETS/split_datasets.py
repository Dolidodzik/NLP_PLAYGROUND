import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('FULL_DATASET.csv', usecols=['review_text', 'rating'])
df = df.dropna(subset=['review_text', 'rating'])

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df.to_csv('TRAINING_DATASET.csv', index=False)
val_df.to_csv('VALIDATION_DATASET.csv', index=False)
test_df.to_csv('TEST_SET.csv', index=False)