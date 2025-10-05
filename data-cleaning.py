import pandas as pd
import ftfy
from ftfy import fix_text

def load_combined_data(path: str = "combined_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
    return df

def remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df_cleaned = df.dropna(subset=["title", "text", "subject"])
    after = len(df_cleaned)
    print(f"🧹 Removed {before - after} rows with missing values (title/text/subject).")
    return df_cleaned

def normalize_date(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'date' column to standard YYYY-MM-DD format."""
    if "date" not in df.columns:
        print("⚠️ No 'date' column found — skipping date normalization.")
        return df

    print("📅 Example raw dates before conversion:")
    print(df["date"].head(10))
    
    # Giữ bản copy của date gốc
    df["date_original"] = df["date"].astype(str).str.strip()
    
    # Format chính: DD-MMM-YY (ví dụ: 17-Feb-18, 19-Jun-16)
    df["date"] = pd.to_datetime(df["date_original"], format="%d-%b-%y", errors="coerce")
    
    failed_count = df["date"].isna().sum()
    
    if failed_count > 0:
        print(f"\n⚠️ {failed_count} rows failed initial conversion. Trying alternative formats...")
        
        failed_mask = df["date"].isna()
        
        # Thử format khác: "July 22, 2017" (tháng đầy đủ + năm 4 chữ số)
        df.loc[failed_mask, "date"] = pd.to_datetime(
            df.loc[failed_mask, "date_original"], 
            format="%B %d, %Y", 
            errors="coerce"
        )
        
        # Thử format: "Jun 19, 2017" (tháng viết tắt + năm 4 chữ số)
        still_failed_mask = df["date"].isna()
        if still_failed_mask.sum() > 0:
            df.loc[still_failed_mask, "date"] = pd.to_datetime(
                df.loc[still_failed_mask, "date_original"], 
                format="%b %d, %Y", 
                errors="coerce"
            )
        
        recovered = failed_count - df["date"].isna().sum()
        print(f"✅ Recovered {recovered} rows with alternative formats")
    
    final_failed = df["date"].isna().sum()
    
    if final_failed > 0:
        print(f"\n⚠️ Final failed count: {final_failed}")
        print("Examples of failed dates:")
        print(df[df["date"].isna()]["date_original"].unique()[:10])
        
        # Drop các dòng không convert được
        before = len(df)
        df = df.dropna(subset=["date"])
        print(f"🗑️ Dropped {before - len(df)} rows with invalid dates")
    
    # Convert sang string format YYYY-MM-DD
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    
    # Xóa cột tạm
    df.drop("date_original", axis=1, inplace=True)
    
    print("\n✅ Date normalization complete!")
    print(f"Final dataset: {len(df)} rows")
    print("\nExample converted dates:")
    print(df["date"].head(10))
    
    return df

def validate_date_format(df: pd.DataFrame) -> pd.DataFrame:
    """Kiểm tra xem cột date có đúng format YYYY-MM-DD không."""
    if "date" not in df.columns:
        print("⚠️ Không tìm thấy cột 'date'")
        return df
    
    print("\n🔍 Đang kiểm tra format cột date...")
    
    # Regex pattern cho YYYY-MM-DD
    date_pattern = r'^\d{4}-\d{2}-\d{2}$'
    
    # Kiểm tra format
    valid_mask = df["date"].astype(str).str.match(date_pattern, na=False)
    valid_count = valid_mask.sum()
    invalid_count = (~valid_mask).sum()
    total_count = len(df)
    
    # Thống kê tổng quan
    print(f"\n📊 THỐNG KÊ:")
    print(f"   ✅ Hợp lệ (YYYY-MM-DD): {valid_count} dòng ({valid_count/total_count*100:.2f}%)")
    print(f"   ❌ Không hợp lệ: {invalid_count} dòng ({invalid_count/total_count*100:.2f}%)")
    print(f"   📝 Tổng: {total_count} dòng")
    
    # In ra các dòng hợp lệ (sample)
    if valid_count > 0:
        print(f"\n✅ MẪU CÁC DÒNG HỢP LỆ (10 dòng đầu):")
        valid_dates = df[valid_mask]["date"].head(10)
        for idx, date in valid_dates.items():
            print(f"   Row {idx}: {date}")
    
    # In ra các dòng không hợp lệ (tất cả nếu ít, hoặc 20 dòng đầu)
    if invalid_count > 0:
        print(f"\n❌ CÁC DÒNG KHÔNG HỢP LỆ:")
        invalid_df = df[~valid_mask][["date"]].copy()
        invalid_df = invalid_df.reset_index()
        
        display_count = min(invalid_count, 20)
        print(f"   (Hiển thị {display_count}/{invalid_count} dòng)\n")
        
        for i, row in invalid_df.head(display_count).iterrows():
            print(f"   Row {row['index']}: {row['date']}")
        
        if invalid_count > 20:
            print(f"\n   ... và {invalid_count - 20} dòng khác")
    else:
        print("\n🎉 TẤT CẢ DATES ĐỀU HỢP LỆ!")
    
    return df

def check_duplicate_title_text(df: pd.DataFrame) -> int:
    """Check how many duplicate rows have the same title and text."""
    duplicate_rows = df[df.duplicated(subset=["title", "text"], keep=False)]
    count = len(duplicate_rows)
    
    print(f"🔍 Found {count} rows with duplicated (title + text).")
    
    # Nếu muốn xem thử vài dòng mẫu
    if count > 0:
        print("📄 Example duplicates:")
        print(duplicate_rows.head(5))
    
    return count

def remove_duplicate_title_text(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows based on title + text combination."""
    before = len(df)
    df = df.drop_duplicates(subset=["title", "text"], keep="first").reset_index(drop=True)
    after = len(df)
    removed = before - after
    
    print(f"🧹 Removed {removed} duplicate rows based on (title + text).")
    print(f"✅ Cleaned dataset now has {after} rows.")
    return df

def fix_encoding_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Sửa lỗi ký tự encoding (ví dụ: â€™, â€œ, Ã©) trong title và text."""
    from ftfy import fix_text

    for col in ["title", "text"]:
        if col in df.columns:
            before_examples = df[col].head(3).tolist()
            print(f"\n🔧 Fixing encoding in '{col}' column...")
            print("Before:")
            for i, ex in enumerate(before_examples):
                print(f"  {i+1}. {ex[:120]}")

            df[col] = df[col].astype(str).apply(fix_text)

            after_examples = df[col].head(3).tolist()
            print("\nAfter:")
            for i, ex in enumerate(after_examples):
                print(f"  {i+1}. {ex[:120]}")
        else:
            print(f"⚠️ Column '{col}' not found, skipping.")
    
    print("\n✅ Encoding issues fixed successfully!")
    return df


def export_to_csv(df: pd.DataFrame, path: str = "combined_dataset_cleaned.csv"):
    """Lưu DataFrame ra file CSV."""
    try:
        df.to_csv(path, index=False, encoding='utf-8')
        print(f"\n💾 Đã lưu thành công DataFrame đã xử lý vào file '{path}'")
    except Exception as e:
        print(f"\n❌ Đã có lỗi xảy ra khi lưu file: {e}")


def main():
    # 1️⃣ Load data
    df = load_combined_data()

    # 2️⃣ Remove missing values
    df = remove_missing_values(df)

    df = fix_encoding_issues(df)

    df = normalize_date(df)

    validate_date_format(df)

    check_duplicate_title_text(df)

    df = remove_duplicate_title_text(df)

    export_to_csv(df)

    
if __name__ == "__main__":
    main()