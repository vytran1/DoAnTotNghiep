import joblib
import shap
import numpy as np
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------- Cấu hình -----------------
MODEL_PATH = 'final_model.joblib'
VECTORIZER_PATH = 'final_tfidf_vectorizer.joblib'
N_BACKGROUND = 50     # số cụm background cho SHAP
N_TEST_SAMPLE = 100   # số mẫu test lấy để tính SHAP
NSAMPLES = 100        # số nsamples cho KernelExplainer

# Tắt cảnh báo
warnings.filterwarnings('ignore', category=UserWarning)

# ----------------- Task 1: Load model, vectorizer, dữ liệu -----------------
print("--- Task 1: Load model, vectorizer, dữ liệu ---")
loaded_model = joblib.load(MODEL_PATH)
loaded_vectorizer = joblib.load(VECTORIZER_PATH)
print(f"Model: {loaded_model.__class__.__name__}")
print("Vectorizer loaded.")

# Load dataset
df = pd.read_csv('./data/processed/data_for_tfidf_merged.csv')
df.dropna(inplace=True)
X_full = df['text_tfidf']
y_full = df['label']

# Split train/test giống lúc train model
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

# Transform text -> TF-IDF
X_train_tfidf = loaded_vectorizer.transform(X_train)
X_test_tfidf = loaded_vectorizer.transform(X_test)
feature_names = loaded_vectorizer.get_feature_names_out()
print(f"Số lượng features: {len(feature_names)}")

# ----------------- Task 2: Tạo background data -----------------
print("--- Task 2: Tạo background data ---")
background_data = shap.kmeans(X_train_tfidf.toarray(), N_BACKGROUND)
print("Background data xong.")

# ----------------- Task 3: Khởi tạo KernelExplainer -----------------
print("--- Task 3: Khởi tạo KernelExplainer ---")
def predict_proba_fn(X):
    if issparse(X):
        X = X.toarray()
    return loaded_model.predict_proba(X)

explainer = shap.KernelExplainer(predict_proba_fn, background_data)
print("Explainer ready.")

# ----------------- Task 4: Tính SHAP toàn cục -----------------
print("--- Task 4: Tính toán SHAP toàn cục ---")
X_test_sample = shap.sample(X_test_tfidf, N_TEST_SAMPLE)
X_test_sample_dense = X_test_sample.toarray() if issparse(X_test_sample) else np.array(X_test_sample)

shap_values = explainer.shap_values(X_test_sample_dense, nsamples=NSAMPLES)
print("Đã tính SHAP values.")

# Vẽ summary plot cho class 'Real' (1)
plt.figure()
shap.summary_plot(shap_values[1], X_test_sample_dense, feature_names=feature_names,
                  plot_type="bar", max_display=20, show=False)
plt.title("Top 20 từ ảnh hưởng đến dự đoán 'Tin THẬT'")
plt.tight_layout()
plt.show()

# Vẽ summary plot cho class 'Fake' (0)
plt.figure()
shap.summary_plot(shap_values[0], X_test_sample_dense, feature_names=feature_names,
                  plot_type="bar", max_display=20, show=False)
plt.title("Top 20 từ ảnh hưởng đến dự đoán 'Tin GIẢ'")
plt.tight_layout()
plt.show()

# ----------------- Task 5: SHAP Local Waterfall -----------------
print("--- Task 5: SHAP Local Waterfall ---")
y_pred_test = loaded_model.predict(X_test_tfidf)

# Chọn 1 mẫu Real (1) và 1 mẫu Fake (0) dự đoán đúng
real_idx = np.where((y_test == 1) & (y_pred_test == 1))[0][0]
fake_idx = np.where((y_test == 0) & (y_pred_test == 0))[0][0]

# Mẫu Real
instance_real = X_test_tfidf[real_idx]
instance_real_dense = instance_real.toarray() if issparse(instance_real) else np.array(instance_real)
shap_values_real = explainer.shap_values(instance_real_dense, nsamples=NSAMPLES)
exp_real = shap.Explanation(
    values=shap_values_real[1][0],
    base_values=explainer.expected_value[1],
    data=instance_real_dense[0],
    feature_names=feature_names
)
plt.figure()
shap.plots.waterfall(exp_real, max_display=20, show=False)
plt.title(f"Waterfall mẫu Real idx={real_idx}")
plt.tight_layout()
plt.show()

# Mẫu Fake
instance_fake = X_test_tfidf[fake_idx]
instance_fake_dense = instance_fake.toarray() if issparse(instance_fake) else np.array(instance_fake)
shap_values_fake = explainer.shap_values(instance_fake_dense, nsamples=NSAMPLES)
exp_fake = shap.Explanation(
    values=shap_values_fake[0][0],
    base_values=explainer.expected_value[0],
    data=instance_fake_dense[0],
    feature_names=feature_names
)
plt.figure()
shap.plots.waterfall(exp_fake, max_display=20, show=False)
plt.title(f"Waterfall mẫu Fake idx={fake_idx}")
plt.tight_layout()
plt.show()

print("\n--- SHAP analysis completed ---")
