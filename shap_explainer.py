import joblib
import shap
import numpy as np
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import warnings

# --- Cấu hình ---
MODEL_PATH = 'final_model.joblib'
VECTORIZER_PATH = 'final_tfidf_vectorizer.joblib'
N_BACKGROUND = 50     # số cụm background
N_TEST_SAMPLE = 100   # số mẫu test để tính SHAP
NSAMPLES = 100        # số nsamples cho KernelExplainer

# --- Tắt cảnh báo ---
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Load model, vectorizer, dữ liệu ---
print("--- Task 1: Load model, vectorizer, dữ liệu ---")
loaded_model = joblib.load(MODEL_PATH)
loaded_vectorizer = joblib.load(VECTORIZER_PATH)
print(f"Model: {loaded_model.__class__.__name__}")
print("Vectorizer loaded.")

# Bạn cần chuẩn bị X_train, X_test, y_test giống file train:
# ví dụ load từ CSV
import pandas as pd
df = pd.read_csv('./data/processed/data_for_tfidf_merged.csv')
df.dropna(inplace=True)
X_full = df['text_tfidf']
y_full = df['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

X_train_tfidf = loaded_vectorizer.transform(X_train)
X_test_tfidf = loaded_vectorizer.transform(X_test)
feature_names = loaded_vectorizer.get_feature_names_out()
print(f"Số lượng features: {len(feature_names)}")

# --- 2. Tạo background data ---
print("--- Task 2: Tạo background data ---")
background_data = shap.kmeans(X_train_tfidf.toarray(), N_BACKGROUND)
print("Background data xong.")

# --- 3. Khởi tạo explainer ---
print("--- Task 3: Khởi tạo KernelExplainer ---")
def predict_proba_fn(X):
    if issparse(X):
        X = X.toarray()
    return loaded_model.predict_proba(X)

explainer = shap.KernelExplainer(predict_proba_fn, background_data)
print("Explainer ready.")

# --- 4. Tính SHAP toàn cục ---
print("--- Task 4: Tính toán SHAP toàn cục ---")
X_test_sample = shap.sample(X_test_tfidf, N_TEST_SAMPLE)
X_test_sample_dense = X_test_sample.toarray() if hasattr(X_test_sample, "toarray") else np.array(X_test_sample)
shap_values = explainer.shap_values(X_test_sample_dense, nsamples=NSAMPLES)
print("Đã tính SHAP values.")

# Vẽ summary plot
plt.figure()
shap.summary_plot(shap_values[1], X_test_sample_dense, feature_names=feature_names,
                  plot_type="bar", max_display=20, show=False)
plt.title("Top 20 từ ảnh hưởng 'Tin THẬT'")
plt.tight_layout()
plt.show()

plt.figure()
shap.summary_plot(shap_values[0], X_test_sample_dense, feature_names=feature_names,
                  plot_type="bar", max_display=20, show=False)
plt.title("Top 20 từ ảnh hưởng 'Tin GIẢ'")
plt.tight_layout()
plt.show()

# --- 5. SHAP cục bộ (Waterfall) ---
print("--- Task 5: SHAP Local Waterfall ---")
y_pred_test = loaded_model.predict(X_test_tfidf)

real_idx = np.where((y_test == 1) & (y_pred_test == 1))[0][0]
fake_idx = np.where((y_test == 0) & (y_pred_test == 0))[0][0]

# Mẫu Real
instance_real = X_test_tfidf[real_idx]
shap_values_real = explainer.shap_values(instance_real, nsamples=NSAMPLES)
exp_real = shap.Explanation(
    values=shap_values_real[1][0],
    base_values=explainer.expected_value[1],
    data=instance_real.toarray()[0],
    feature_names=feature_names
)
plt.figure()
shap.plots.waterfall(exp_real, max_display=20, show=False)
plt.title(f"Waterfall mẫu Real idx={real_idx}")
plt.tight_layout()
plt.show()

# Mẫu Fake
instance_fake = X_test_tfidf[fake_idx]
shap_values_fake = explainer.shap_values(instance_fake, nsamples=NSAMPLES)
exp_fake = shap.Explanation(
    values=shap_values_fake[0][0],
    base_values=explainer.expected_value[0],
    data=instance_fake.toarray()[0],
    feature_names=feature_names
)
plt.figure()
shap.plots.waterfall(exp_fake, max_display=20, show=False)
plt.title(f"Waterfall mẫu Fake idx={fake_idx}")
plt.tight_layout()
plt.show()

print("\n--- SHAP analysis completed ---")