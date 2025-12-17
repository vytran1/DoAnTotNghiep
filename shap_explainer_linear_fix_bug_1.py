import os
import joblib
import shap
import numpy as np
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split

# ==================== CONFIG ====================
MODEL_PATH = 'final_model_full_data.joblib'
VECTORIZER_PATH = 'final_tfidf_vectorizer_full_data.joblib'
DATA_PATH = './data_for_tfidf_merged.csv'

N_TEST_SAMPLE = 500    # số mẫu test lấy để tính SHAP global

warnings.filterwarnings('ignore', category=UserWarning)

print("="*70)
print("             SHAP ANALYSIS – LOGISTIC REGRESSION (Corrected)")
print("="*70)

# ==================== TASK 1: LOAD DATA ====================
print("\n[1/5] 📂 Loading model, vectorizer và dataset...")

try:
    loaded_model = joblib.load(MODEL_PATH)
    loaded_vectorizer = joblib.load(VECTORIZER_PATH)

    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)

except FileNotFoundError as e:
    print(f"LỖI: Không tìm thấy file. Hãy chắc chắn file sau có tồn tại: {e.filename}")
    exit()
except Exception as e:
    print(f"Lỗi khi tải file: {e}")
    exit()

print(f"    ✓ Model: {loaded_model.__class__.__name__}")
print(f"    ✓ Vectorizer loaded")
print(f"    ✓ Dataset loaded: {len(df)} samples")

X_full = df['text_tfidf']
y_full = df['label']

# Tái tạo train/test split y hệt như lúc training
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

X_train_tfidf = loaded_vectorizer.transform(X_train)
X_test_tfidf = loaded_vectorizer.transform(X_test)
feature_names = loaded_vectorizer.get_feature_names_out()

print(f"    ✓ Train: {X_train_tfidf.shape}, Test: {X_test_tfidf.shape}")
print(f"    ✓ Features: {len(feature_names)}")


# ==================== TASK 2: INIT EXPLAINER ====================
print("\n[2/5] ⚡ Khởi tạo LinearExplainer (chuẩn cho Logistic Regression)...")

explainer = shap.LinearExplainer(
    loaded_model,
    X_train_tfidf,
    feature_perturbation="interventional"
)

print("    ✓ LinearExplainer ready!")


# ==================== TASK 3: GLOBAL SHAP ====================
print(f"\n[3/5] 🧮 Tính SHAP cho {N_TEST_SAMPLE} mẫu test...")

# Lấy mẫu từ tập test để chạy nhanh hơn
X_test_sample = shap.sample(X_test_tfidf, min(N_TEST_SAMPLE, X_test_tfidf.shape[0]))
shap_values = explainer.shap_values(X_test_sample)

print(f"    ✓ SHAP values shape: {shap_values.shape}")


# ==================== GLOBAL PLOTS ====================
print("\n[4/5] 📊 Vẽ Global SHAP (Top 20 Features)...")

mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_abs_shap)[-20:][::-1]

top_features = feature_names[top_idx]
top_values = mean_abs_shap[top_idx]

plt.figure(figsize=(12, 8))
bars = plt.barh(range(len(top_features)), top_values, color='purple', alpha=0.8)
plt.yticks(range(len(top_features)), top_features)
plt.xlabel("Mean |SHAP value| (Ảnh hưởng đến dự đoán Class 1)")
plt.title("Top 20 Features Quan Trọng Nhất (Global SHAP)")
plt.gca().invert_yaxis()

for i, v in enumerate(top_values):
    plt.text(v + 0.001, i, f"{v:.4f}", va='center')

plt.tight_layout()
plt.savefig("shap_global_top20.png", dpi=300, bbox_inches='tight')
plt.show()

print("    ✓ Đã lưu: shap_global_top20.png")

print("    Vẽ Beeswarm plot...")

X_sample_dense = X_test_sample.toarray() if issparse(X_test_sample) else X_test_sample

shap.summary_plot(
    shap_values,
    X_sample_dense,
    feature_names=feature_names,
    max_display=20,
    show=False
)

plt.tight_layout()
plt.savefig("shap_global_beeswarm.png", dpi=300, bbox_inches='tight')
plt.show()

print("    ✓ Đã lưu: shap_global_beeswarm.png")


# ==================== TASK 5: LOCAL SHAP – WATERFALL ====================
print("\n[5/5] 🌊 Vẽ Waterfall Plot cho mẫu cụ thể...")

y_pred = loaded_model.predict(X_test_tfidf)
y_test_array = y_test.values

# Tìm một mẫu True Positive (dự đoán 1, thực tế 1)
real_idx_all = np.where((y_test_array == 1) & (y_pred == 1))[0]
if len(real_idx_all) == 0:
    print("KHÔNG TÌM THẤY MẪU TRUE POSITIVE. Bỏ qua waterfall cho REAL.")
    real_idx = -1
else:
    real_idx = real_idx_all[0] # Lấy mẫu đầu tiên

# Tìm một mẫu True Negative (dự đoán 0, thực tế 0)
fake_idx_all = np.where((y_test_array == 0) & (y_pred == 0))[0]
if len(fake_idx_all) == 0:
    print("KHÔNG TÌM THẤY MẪU TRUE NEGATIVE. Bỏ qua waterfall cho FAKE.")
    fake_idx = -1
else:
    fake_idx = fake_idx_all[0] # Lấy mẫu đầu tiên


# ========= REAL NEWS (class 1) =========
if real_idx != -1:
    print(f"\n    ✅ Mẫu REAL (idx={real_idx}) - Giải thích cho Class 1")

    instance_real = X_test_tfidf[real_idx]
    instance_real_dense = instance_real.toarray().flatten()

    # SỬA LỖI: Lấy [0] vì shap_values chỉ trả về 1 array (1, n_features)
    shap_real = explainer.shap_values(instance_real)[0]
    # SỬA LỖI: expected_value là 1 float, không phải list
    base_real = explainer.expected_value

    exp_real = shap.Explanation(
        values=shap_real,
        base_values=base_real,
        data=instance_real_dense,
        feature_names=feature_names
    )

    plt.figure(figsize=(12, 8))
    # Dùng .plot vì shap.plots.waterfall có thể bị lỗi layout
    shap.plots.waterfall(exp_real, max_display=20, show=False)
    plt.title(f"Waterfall Plot – REAL NEWS (idx={real_idx}) - Dự đoán Class 1")
    plt.tight_layout()
    plt.savefig(f"shap_waterfall_real_{real_idx}.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"    ✓ Đã lưu: shap_waterfall_real_{real_idx}.png")


# ========= FAKE NEWS (class 0) =========
if fake_idx != -1:
    print(f"\n    ✅ Mẫu FAKE (idx={fake_idx}) - Giải thích cho Class 0")

    instance_fake = X_test_tfidf[fake_idx]
    instance_fake_dense = instance_fake.toarray().flatten()

    # SỬA LỖI: SHAP cho class 0 là ÂM BẢN của SHAP cho class 1
    shap_fake = -explainer.shap_values(instance_fake)[0]
    # SỬA LỖI: Base value cho class 0 là ÂM BẢN của base value cho class 1
    base_fake = -explainer.expected_value

    exp_fake = shap.Explanation(
        values=shap_fake,
        base_values=base_fake,
        data=instance_fake_dense,
        feature_names=feature_names
    )

    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(exp_fake, max_display=20, show=False)
    plt.title(f"Waterfall Plot – FAKE NEWS (idx={fake_idx}) - Dự đoán Class 0")
    plt.tight_layout()
    plt.savefig(f"shap_waterfall_fake_{fake_idx}.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"    ✓ Đã lưu: shap_waterfall_fake_{fake_idx}.png")


# ==================== DONE ====================
print("\n" + "="*70)
print("                       ✅ HOÀN TẤT!")
print("="*70)
print(f"📊 Đã phân tích: {X_test_sample.shape[0]} mẫu test")
print(f"🔤 Số features: {len(feature_names)}")
print(f"💾 Lưu 4 file:")
print("    • shap_global_top20.png")
print("    • shap_global_beeswarm.png")
if real_idx != -1:
    print(f"    • shap_waterfall_real_{real_idx}.png")
if fake_idx != -1:
    print(f"    • shap_waterfall_fake_{fake_idx}.png")
print(f"⚡ LinearExplainer chạy ~100+ lần nhanh hơn KernelExplainer")
print("="*70)