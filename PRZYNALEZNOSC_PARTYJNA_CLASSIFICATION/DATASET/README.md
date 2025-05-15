# list of steps needed to produce dataset

# dataset_analysis.py is just support file, does nothing concrete, is used to interact with datasets and test shit

1. dataset_analysis.py outputs "FULL_DATASET_ORIGINAL.csv" - full, raw dataset, with basic cleaning
2. dataset_cleaning.py - cleans stuff, such as (Oklaski!) - takes "FULL_DATASET_ORIGINAL.csv" as an input, and outputs "FULL_DATASET_CLEANED.csv"
3. dataset_splitting.py - takes N as an input natural number, splits "FULL_DATASET_CLEANED.csv" and takes random sample of N elements of each club (if some club doesn't have at least N statements, then it will put in all its rows). Then, it randomly splits the data into 80/20, training/value datasets - "DATASET_TRAINING.csv" and "DATASET_VALIDATION.csv".
4. data_augumentation.py - takes "DATASET_TRAINING.csv" - and does data augumentation - finds N by checking for the club(s) with most statements, and for the clubs that have less than N statements to their name, does data augumentation on their existing statements, until each club has exactly N statements.