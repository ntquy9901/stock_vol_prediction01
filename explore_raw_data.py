import pandas as pd

# Load raw dataset
df = pd.read_excel('temp_github_dataset/Dataset/raw_data.xlsx')

print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst 3 rows:\n{df.head(3)}")
print(f"\nData types:\n{df.dtypes}")

# Check label distribution if exists
if len(df.columns) > 1:
    print(f"\nUnique labels in last column:")
    print(df.iloc[:, -1].value_counts())
