import pandas as pd

dataset = pd.read_csv('FULL_DATASET_ORIGINAL_2024_TO_MAY_2025.csv')

# removing paranthesies and text inside them - stuff like (Oklaski) that is often present in our dataset
dataset['statement'] = dataset['statement'].str.replace(r'\([^)]*\)', '', regex=True)

# getting rid of unneeded white space, and converts everything to lower-case - letter case would extremely rarely matter in our task
dataset['statement'] = dataset['statement'].str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()

dataset.to_csv('CLEANED_DATASET.csv', index=False)