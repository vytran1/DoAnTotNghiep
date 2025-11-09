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
import joblib 
import shap
import warnings
from scipy.sparse import issparse

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

    df = pd.read_csv('./data/processed/data_for_tfidf_merged.csv')
    df.dropna(inplace=True)
except FileNotFoundError:
    print("File not found! Make sure 'data_for_tfidf_merged.csv' is in the correct path.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()


X_full = df['text_tfidf']
y_full = df['label']


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
tfidf_selector = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = tfidf_selector.fit_transform(X_train)
X_test_tfidf = tfidf_selector.transform(X_test) 

print(f"TF-IDF training vectors shape: {X_train_tfidf.shape}")
print(f"TF-IDF testing vectors shape: {X_test_tfidf.shape}")
print("-" * 30)

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

best_models_evaluation = {}

print("\n--- 4. Chạy GridSearchCV cho từng mô hình ---")
for model_name, config in models_and_params.items():
    print(f"\n--- Bắt đầu Training và Đánh giá {model_name} ---")

    grid_search = GridSearchCV(
        config['model'],
        config['params'],
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_tfidf, y_train)

    
    print("\n--- Chi tiết kết quả GridSearchCV ---")
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cols_to_show = ['param_C', 'param_solver', 'param_kernel', 'param_alpha', 'param_max_iter', 'mean_test_score', 'std_test_score', 'rank_test_score']
    existing_cols = [col for col in cols_to_show if col in cv_results_df.columns]
    print(cv_results_df[existing_cols].sort_values(by='rank_test_score'))
    print("-" * 20)

    print(f"Best parameters for {model_name}: {grid_search.best_params_}")

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

    best_models_evaluation[model_name] = {
        'model_class': config['model'].__class__,
        'best_params': grid_search.best_params_,
        'test_f1': f1,
        'test_accuracy': accuracy
    }
    print("-" * 30)


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

print("\n--- 6. Train lại mô hình tốt nhất trên TOÀN BỘ dataset ---")

print("Fitting TF-IDF trên toàn bộ dữ liệu...")
final_tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
X_full_tfidf = final_tfidf_vectorizer.fit_transform(X_full)
print(f"Kích thước TF-IDF cuối cùng: {X_full_tfidf.shape}")

print(f"Training mô hình {best_overall_model_name}...")
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

print("\n--- 7. Lưu mô hình và vectorizer ---")
model_path = 'final_model.joblib'
vectorizer_path = 'final_tfidf_vectorizer.joblib'

joblib.dump(final_model, model_path)
joblib.dump(final_tfidf_vectorizer, vectorizer_path)

print(f"Mô hình đã lưu tại: {model_path}")
print(f"Vectorizer đã lưu tại: {vectorizer_path}")

print("\n--- 8. Ví dụ sử dụng mô hình đã lưu ---")
loaded_model = joblib.load(model_path)
loaded_vectorizer = joblib.load(vectorizer_path)

sample_texts = [
    "Đây là một ví dụ về tin tức thật", 
    "Tin này hoàn toàn bịa đặt" 
]

sample_tfidf = loaded_vectorizer.transform(sample_texts)
predictions = loaded_model.predict(sample_tfidf)
probabilities = loaded_model.predict_proba(sample_tfidf)

print(f"Văn bản mẫu: {sample_texts[0]} -> Dự đoán: {predictions[0]}, Xác suất: {probabilities[0]}")
print(f"Văn bản mẫu: {sample_texts[1]} -> Dự đoán: {predictions[1]}, Xác suất: {probabilities[1]}")

print("\n--- Hoàn tất quy trình ---")

print(f"Mô hình đã lưu tại: {model_path}")
print(f"Vectorizer đã lưu tại: {vectorizer_path}")