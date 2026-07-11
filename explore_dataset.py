import pandas as pd

# Load dataset
df = pd.read_excel('temp_github_dataset/Dataset/Macp.xlsx')

print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")

# Check label distribution
if len(df.columns) > 0:
    print(f"\nLabel distribution:\n{df.iloc[:, -1].value_counts()}")
