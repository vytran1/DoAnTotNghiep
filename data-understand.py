import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_csv("train.tsv", sep="\t")
df_val = pd.read_csv("validation.tsv", sep="\t")
df_test = pd.read_csv("test.tsv", sep="\t")

df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
#Kiểm tra số lượng dòng, cột, kiểu dữ liệu (title, text, subject, date).
# print("📊 Total number of rows:", len(df_all))
# print("📋 Columns:", df_all.columns.tolist())
# print("\n📦 Data types:")
# print(df_all.dtypes)
# print("\n🔍 First 5 rows:")
# print(df_all.head())
# print("\n🚫 Number of missing (null) values per column:")
# print(df_all.isnull().sum())

#Thống kê, vẽ biểu đồ 
print("\n📰 Label distribution:")
print(df_all['label'].value_counts())
print("\n📈 Label percentage:")
print(df_all['label'].value_counts(normalize=True) * 100)
print("\n📚 Number of subjects:", df_all['subject'].nunique())
print(df_all['subject'].value_counts().head(10))

#Biểu Đồ phân bổ fake and real news
# df_all['label'].value_counts().plot(
#     kind='bar',
#     color=['tomato', 'skyblue'],
#     title='Distribution of Fake (0) vs Real (1) News'
# )
# plt.xlabel('Label')
# plt.ylabel('Count')
# plt.show()

#Biểu Đồ phân bổ theo chủ đề
df_all['subject'].value_counts().head(10).plot(
    kind='bar',
    figsize=(8,4),
    title='Top 10 Subjects in Dataset'
)
plt.xlabel('Subject')
plt.ylabel('Count')
plt.show()

subject_label_counts = df_all.groupby(['subject', 'label']).size().unstack(fill_value=0)
print(subject_label_counts)


subject_label_counts.plot(
    kind='bar',
    figsize=(10,6),
    color=['tomato', 'skyblue'],
    width=0.7
)

plt.title('Fake vs Real News by Subject')
plt.xlabel('Subject')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45, ha='right')
plt.legend(['Fake (0)', 'Real (1)'])
plt.tight_layout()
plt.show()