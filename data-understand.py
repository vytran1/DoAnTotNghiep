import pandas as pd
import matplotlib.pyplot as plt


# 🧩 1️⃣ Load dataset
def load_data(train_path: str, test_path: str) -> pd.DataFrame:
    """Load train and test datasets, then merge them."""
    df_train = pd.read_csv(train_path, sep="\t")
    df_test = pd.read_csv(test_path, sep="\t")
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    return df_all


# 🧩 2️⃣ Basic info
def show_basic_info(df: pd.DataFrame):
    """Print dataset overview: rows, columns, dtypes, and null counts."""
    print("📊 Total number of rows:", len(df))
    print("📋 Columns:", df.columns.tolist())
    print("\n📦 Data types:")
    print(df.dtypes)
    print("\n🚫 Missing values per column:")
    print(df.isnull().sum())


# 🧩 3️⃣ Label distribution
def plot_label_distribution(df: pd.DataFrame):
    """Show count and percentage of Fake vs Real news."""
    print("\n📰 Label distribution:")
    print(df['label'].value_counts())
    print("\n📈 Label percentage:")
    print(df['label'].value_counts(normalize=True) * 100)

    df['label'].value_counts().plot(
        kind='bar',
        color=['tomato', 'skyblue'],
        title='Distribution of Fake (0) vs Real (1) News'
    )
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()


# 🧩 4️⃣ Subject analysis
def plot_subject_distribution(df: pd.DataFrame):
    """Show top subjects and Fake vs Real counts per subject."""
    print("\n📚 Number of subjects:", df['subject'].nunique())
    print(df['subject'].value_counts().head(10))

    # Plot top 10 subjects
    df['subject'].value_counts().head(10).plot(
        kind='bar',
        figsize=(8,4),
        title='Top 10 Subjects in Dataset'
    )
    plt.xlabel('Subject')
    plt.ylabel('Count')
    plt.show()

    # Group by subject and label
    subject_label_counts = df.groupby(['subject', 'label']).size().unstack(fill_value=0)
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


# 🧩 5️⃣ Missing and duplicate data check
def check_missing_and_duplicates(df: pd.DataFrame):
    """Check missing values and duplicates."""
    print("🔍 Missing values per column:")
    print(df.isnull().sum())

    print(f"\n🧩 Number of duplicate rows: {df.duplicated().sum()}")
    print(f"📰 Duplicate titles: {df['title'].duplicated().sum()}")
    print(f"📝 Duplicate texts: {df['text'].duplicated().sum()}")


# 🧩 6️⃣ Main pipeline
def main():
    df = load_data("train.tsv", "test.tsv")

    # Basic dataset info
    show_basic_info(df)

    # Label and subject distributions
    plot_label_distribution(df)
    plot_subject_distribution(df)

    # Missing and duplicate data
    check_missing_and_duplicates(df)


if __name__ == "__main__":
    main()
