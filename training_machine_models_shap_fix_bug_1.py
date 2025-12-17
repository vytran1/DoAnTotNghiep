import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib # Thêm thư viện để lưu model
import shap
import warnings 
from scipy.sparse import issparse 

# --- 0. Hàm tiện ích ---
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Hàm này in và vẽ confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

print("--- 1. Tải và Chuẩn bị Dữ liệu ---")
try:
    # Sửa tên file cho đúng với mô tả của bạn
    df = pd.read_csv('./data/processed/data_for_tfidf_merged.csv') 
    df.dropna(inplace=True)
except FileNotFoundError:
    print("File not found! Make sure 'data_for_tfidf_merged.csv' is in the correct path.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# Đây là toàn bộ dataset
X_full = df['text_tfidf']
y_full = df['label']

# Chia train/test để *đánh giá* và *chọn* siêu tham số (hyperparameters)
X_train, X_test, y_train, y_test = train_test_split(
    X_full,
    y_full,
    test_size=0.2,
    random_state=42,
    stratify=y_full
)
print(f"Tổng số mẫu: {len(X_full)}")
print(f"Dữ liệu training: {X_train.shape}")
print(f"Dữ liệu testing: {X_test.shape}")
print("-" * 30)


print("\n--- 2. Tạo TF-IDF Vectors (chỉ từ tập Train) ---")
# Chỉ fit trên tập train để tránh rò rỉ dữ liệu (data leakage)
tfidf_selector = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = tfidf_selector.fit_transform(X_train)
X_test_tfidf = tfidf_selector.transform(X_test) # Chỉ transform tập test

print(f"TF-IDF training vectors shape: {X_train_tfidf.shape}")
print(f"TF-IDF testing vectors shape: {X_test_tfidf.shape}")
print("-" * 30)

# --- 3. Định nghĩa các mô hình và tham số để thử nghiệm ---
models_and_params = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'saga']
        }
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42),
        'params': {
            'C': [1, 10], 
            'kernel': ['linear'] 
        }
    },
    'Naive Bayes': {
        'model': MultinomialNB(),
        'params': {
            'alpha': [0.1, 0.5, 1.0]
        }
    }
}

# Lưu trữ kết quả tốt nhất của mỗi mô hình
best_models_evaluation = {}

print("\n--- 4. Chạy GridSearchCV cho từng mô hình ---")
for model_name, config in models_and_params.items():
    print(f"\n--- Bắt đầu Training và Đánh giá {model_name} ---")

    grid_search = GridSearchCV(
        config['model'],
        config['params'],
        cv=5,
        scoring='accuracy', # Bạn có thể đổi sang 'f1' nếu muốn
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_tfidf, y_train)

    # --- YÊU CẦU 1: Hiển thị kết quả chi tiết của từng bộ tham số ---
    print("\n--- Chi tiết kết quả GridSearchCV ---")
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    # Sắp xếp theo rank và chỉ lấy các cột quan trọng
    cols_to_show = ['param_C', 'param_solver', 'param_kernel', 'param_alpha', 'param_max_iter', 'mean_test_score', 'std_test_score', 'rank_test_score']
    # Lọc ra các cột thực sự có trong kết quả
    existing_cols = [col for col in cols_to_show if col in cv_results_df.columns]
    print(cv_results_df[existing_cols].sort_values(by='rank_test_score'))
    print("-" * 20)

    # --- YÊU CẦU 2: Hiển thị tham số tốt nhất ---
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")

    # Đánh giá mô hình tốt nhất (từ grid search) trên tập TEST
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_tfidf)
    y_pred_proba = best_model.predict_proba(X_test_tfidf)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("\nĐánh giá trên TẬP TEST (dùng để chọn mô hình):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")

    print("\nClassification Report (trên tập Test):")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=['Fake (0)', 'Real (1)'], title=f'{model_name} Confusion Matrix (Test Set)')
    
    # Lưu lại kết quả để so sánh
    best_models_evaluation[model_name] = {
        'model_class': config['model'].__class__, # Lấy class của model
        'best_params': grid_search.best_params_,
        'test_f1': f1, # Dùng F1-score để chọn mô hình cuối cùng
        'test_accuracy': accuracy
    }
    print("-" * 30)

# --- YÊU CẦU 3: Chọn mô hình tốt nhất chung cuộc và Train lại trên toàn bộ data ---

print("\n--- 5. So sánh các mô hình và chọn ra mô hình tốt nhất ---")
print("Tổng kết hiệu suất trên tập Test:")
best_overall_model_name = None
best_overall_f1 = -1

for model_name, metrics in best_models_evaluation.items():
    print(f"  {model_name}: Test F1 = {metrics['test_f1']:.4f}, Test Accuracy = {metrics['test_accuracy']:.4f}")
    if metrics['test_f1'] > best_overall_f1:
        best_overall_f1 = metrics['test_f1']
        best_overall_model_name = model_name

print(f"\n=> Mô hình tốt nhất chung cuộc (dựa trên F1-score): {best_overall_model_name}")

best_config = best_models_evaluation[best_overall_model_name]
final_model_class = best_config['model_class']
final_model_params = best_config['best_params']

print(f"Sẽ train lại mô hình {best_overall_model_name} với tham số {final_model_params} trên TOÀN BỘ dataset.")

# --- Train lại ---
print("\n--- 6. Train lại mô hình tốt nhất trên TOÀN BỘ dataset ---")

# 1. Tạo và fit TF-IDF trên toàn bộ X_full
print("Fitting TF-IDF trên toàn bộ dữ liệu...")
final_tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
X_full_tfidf = final_tfidf_vectorizer.fit_transform(X_full)
print(f"Kích thước TF-IDF cuối cùng: {X_full_tfidf.shape}")

# 2. Khởi tạo và train mô hình cuối cùng
print(f"Training mô hình {best_overall_model_name}...")
# Cần xử lý đặc biệt cho các model có tham số không giống tên
if final_model_class == LogisticRegression:
    final_model = LogisticRegression(random_state=42, **final_model_params)
elif final_model_class == SVC:
    final_model = SVC(probability=True, random_state=42, **final_model_params)
elif final_model_class == MultinomialNB:
    final_model = MultinomialNB(**final_model_params)
elif final_model_class == SGDClassifier:
    final_model = SGDClassifier(loss='log_loss', random_state=42, **final_model_params)

final_model.fit(X_full_tfidf, y_full)

print("Đã train xong mô hình cuối cùng!")

# --- YÊU CẦU 4: Lưu lại mô hình ---
print("\n--- 7. Lưu mô hình và vectorizer ---")
model_path = 'final_model.joblib'
vectorizer_path = 'final_tfidf_vectorizer.joblib'

joblib.dump(final_model, model_path)
joblib.dump(final_tfidf_vectorizer, vectorizer_path)

print(f"Mô hình đã lưu tại: {model_path}")
print(f"Vectorizer đã lưu tại: {vectorizer_path}")

print("\n--- 8. Ví dụ sử dụng mô hình đã lưu ---")
# Tải lại
loaded_model = joblib.load(model_path)
loaded_vectorizer = joblib.load(vectorizer_path)

# Dữ liệu mới (ví dụ)
sample_texts = [
    "Đây là một ví dụ về tin tức thật", # Giả sử đây là 1
    "Tin này hoàn toàn bịa đặt" # Giả sử đây là 0
]

# Sử dụng mô hình đã tải
sample_tfidf = loaded_vectorizer.transform(sample_texts)
predictions = loaded_model.predict(sample_tfidf)
probabilities = loaded_model.predict_proba(sample_tfidf)

print(f"Văn bản mẫu: {sample_texts[0]} -> Dự đoán: {predictions[0]}, Xác suất: {probabilities[0]}")
print(f"Văn bản mẫu: {sample_texts[1]} -> Dự đoán: {predictions[1]}, Xác suất: {probabilities[1]}")

print("\n--- Hoàn tất quy trình ---")

print(f"Mô hình đã lưu tại: {model_path}")
print(f"Vectorizer đã lưu tại: {vectorizer_path}")


# --- 8. Giải thích mô hình với SHAP (Thực hiện 5 tasks của bạn) ---
print("\n" + "="*40)
print("--- 8. Giải thích mô hình với SHAP ---")
print("="*40)
# Tắt một số cảnh báo để output sạch hơn
warnings.filterwarnings('ignore', category=UserWarning)

try:
    # --- Task 1: Chuẩn bị Môi trường & Dữ liệu ---
    print("\n--- SHAP Task 1: Tải mô hình, vectorizer và dữ liệu ---")
    loaded_model = joblib.load(model_path)
    loaded_vectorizer = joblib.load(vectorizer_path)
    
    # Chúng ta sử dụng lại X_train, X_test, y_test đã được chia ở Bước 1
    # để giải thích mô hình cuối cùng
    print(f"Đã tải mô hình: {loaded_model.__class__.__name__}")
    print(f"Đã tải vectorizer.")

    # Biến đổi (transform) dữ liệu train và test bằng vectorizer đã lưu
    X_train_tfidf_shap = loaded_vectorizer.transform(X_train)
    X_test_tfidf_shap = loaded_vectorizer.transform(X_test)
    
    # Lấy tên của tất cả các features (từ vựng)
    feature_names = loaded_vectorizer.get_feature_names_out()
    print(f"Số lượng features (từ vựng): {len(feature_names)}")

    # --- Task 2: Tạo Dữ liệu Nền (Background Data) ---
    print("\n--- SHAP Task 2: Tạo dữ liệu nền (background data) ---")
    print("Sử dụng shap.kmeans để tóm tắt 50 cụm từ tập train...")
    # SHAP cần một bộ dữ liệu nền (thường là từ tập train) để so sánh
    # Vì tập train lớn, ta dùng K-Means để tóm tắt nó
    background_data = shap.kmeans(X_train_tfidf_shap.toarray(), 50)
    print("Đã tạo xong dữ liệu nền.")

    # --- Task 3: Khởi tạo Explainer ---
    print("\n--- SHAP Task 3: Khởi tạo SHAP KernelExplainer ---")
    
    # Viết hàm predict_proba wrapper
    # KernelExplainer cần một hàm nhận vào mảng numpy (dense)
    def predict_proba_fn(X):
        # Chuyển đổi sparse matrix sang dense array nếu cần
        if issparse(X):
            X = X.toarray()
        # Trả về xác suất (probability)
        return loaded_model.predict_proba(X)

    # Khởi tạo explainer
    explainer = shap.KernelExplainer(predict_proba_fn, background_data)
    print("Đã khởi tạo Explainer.")

    # --- Task 4: Tính toán & Trực quan hóa (Toàn cục) ---
    print("\n--- SHAP Task 4: Tính toán và vẽ SHAP toàn cục (Global) ---")
    print("Tính toán SHAP values cho 100 mẫu test (việc này CÓ THỂ MẤT VÀI PHÚT)...")
    
    # Lấy 100 mẫu test ngẫu nhiên để tính toán
    # Tính trên toàn bộ X_test (ví dụ 1000 câu) sẽ RẤT LÂU
    X_test_sample_shap = shap.sample(X_test_tfidf_shap, 100)
    
    # Tính SHAP values
    # nsamples='auto' (hoặc 100) để tăng tốc độ
    X_test_sample_shap_dense = X_test_sample_shap.toarray()
    shap_values = explainer.shap_values(X_test_sample_shap_dense, nsamples=100)
    print("Đã tính xong SHAP values.")

    # Trực quan hóa (Lưu ý: Class 1 = 'Real', Class 0 = 'Fake')
    print("\n--- Biểu đồ SHAP toàn cục (Global Summary Plot) ---")
    
    # Giải thích cho Class 1 (Tin THẬT)
    # shap_values[1] là giá trị SHAP cho class 1
    plt.figure()
    shap.summary_plot(shap_values[1], X_test_sample_shap_dense, feature_names=feature_names, plot_type="bar", max_display=20, show=False)
    plt.title("20 từ ảnh hưởng lớn nhất đến dự đoán 'Tin THẬT' (Class 1)")
    plt.tight_layout()
    plt.show()

    # Giải thích cho Class 0 (Tin GIẢ) - (Đây là cái bạn hỏi 'Tin giả index 1'?)
    plt.figure()
    shap.summary_plot(shap_values[0], X_test_sample_shap_dense, feature_names=feature_names, plot_type="bar", max_display=20, show=False)
    plt.title("20 từ ảnh hưởng lớn nhất đến dự đoán 'Tin GIẢ' (Class 0)")
    plt.tight_layout()
    plt.show()

    # --- Task 5: Trực quan hóa (Cục bộ) ---
    print("\n--- SHAP Task 5: Vẽ SHAP cục bộ (Local Waterfall Plot) ---")
    
    # Dự đoán trên toàn bộ tập test để tìm mẫu
    y_pred_test = loaded_model.predict(X_test_tfidf_shap)
    
    # Tìm 1 mẫu dự đoán 'Real' (1) và 1 mẫu 'Fake' (0) (và dự đoán đúng)
    try:
        real_idx = np.where((y_test == 1) & (y_pred_test == 1))[0][0]
        fake_idx = np.where((y_test == 0) & (y_pred_test == 0))[0][0]
    except IndexError:
        print("Không tìm thấy mẫu dự đoán đúng, lấy 2 mẫu bất kỳ")
        real_idx = np.where(y_test == 1)[0][0]
        fake_idx = np.where(y_test == 0)[0][0]

    # --- Giải thích cho 1 mẫu Tin THẬT ---
    print(f"\n--- Giải thích cho mẫu {real_idx} (Dự đoán: THẬT) ---")
    print(f"Văn bản gốc: {X_test.iloc[real_idx][:100]}...")
    
    # Lấy 1 mẫu duy nhất
    instance_real = X_test_tfidf_shap[real_idx]
    
    # Tính SHAP (phải tính lại vì KernelExplainer)
    # expected_value là giá trị dự đoán trung bình trên tập background
    shap_values_real = explainer.shap_values(instance_real, nsamples=100)
    
    # Tạo đối tượng Explanation để vẽ waterfall
    # shap_values_real[1][0] = giá trị SHAP cho class 1, mẫu 0 (chỉ có 1 mẫu)
    # explainer.expected_value[1] = giá trị base cho class 1
    exp_real = shap.Explanation(
        values=shap_values_real[1][0],
        base_values=explainer.expected_value[1],
        data=instance_real.toarray()[0], # Cần dense array
        feature_names=feature_names
    )
    
    plt.figure()
    shap.plots.waterfall(exp_real, max_display=20, show=False)
    plt.title(f"Giải thích cho dự đoán 'Tin THẬT' (Class 1) - Mẫu {real_idx}")
    plt.tight_layout()
    plt.show()

    # --- Giải thích cho 1 mẫu Tin GIẢ ---
    print(f"\n--- Giải thích cho mẫu {fake_idx} (Dự đoán: GIẢ) ---")
    print(f"Văn bản gốc: {X_test.iloc[fake_idx][:100]}...")
    
    instance_fake = X_test_tfidf_shap[fake_idx]
    shap_values_fake = explainer.shap_values(instance_fake, nsamples=100)
    
    # Giải thích cho Class 0 (Tin Giả)
    exp_fake = shap.Explanation(
        values=shap_values_fake[0][0],
        base_values=explainer.expected_value[0],
        data=instance_fake.toarray()[0],
        feature_names=feature_names
    )
    
    plt.figure()
    shap.plots.waterfall(exp_fake, max_display=20, show=False)
    plt.title(f"Giải thích cho dự đoán 'Tin GIẢ' (Class 0) - Mẫu {fake_idx}")
    plt.tight_layout()
    plt.show()

except ImportError:
    print("\nLỖI: Thư viện 'shap' chưa được cài đặt.")
    print("Vui lòng chạy: pip install shap")
except Exception as e:
    print(f"\nĐã xảy ra lỗi trong quá trình chạy SHAP: {e}")


print("\n--- Hoàn tất quy trình (bao gồm cả SHAP) ---")