import pandas as pd

# Load Excel file
df = pd.read_excel("modeldataset.xlsx")

print("Original Shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows where all values are missing
df = df.dropna(how='all')

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Remove extra spaces from text columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Fill missing values
"""
select_dtypes(include='number') → selects all numeric columns.
for col in ... → goes through each numeric column one by one.
df[col].mean() → calculates the average of that column.
fillna(...) → replaces missing values (NaN) with that average.
"""
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].mean())

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nCleaned Shape:", df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Preview cleaned data
print("\nFirst 5 Rows:")
print(df.head())

# Save cleaned dataset
df.to_excel("cleaned_dataset.xlsx", index=False)

print("\nData cleaned and saved successfully as 'cleaned_dataset.xlsx'")