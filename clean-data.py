import pandas as pd
df_train = pd.read_csv("train.tsv", sep="\t")
df_val = pd.read_csv("validation.tsv", sep="\t")
df_test = pd.read_csv("test.tsv", sep="\t")

df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

print("📊 Total number of rows:", len(df_all))
print("📋 Columns:", df_all.columns.tolist())
print("\n📦 Data types:")
print(df_all.dtypes)
print("\n🔍 First 5 rows:")
print(df_all.head())
print("\n🚫 Number of missing (null) values per column:")
print(df_all.isnull().sum())