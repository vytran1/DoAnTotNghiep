import joblib
import shap
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse
import warnings

# ==================== CONFIG ====================
MODEL_PATH = '../model/final_model_full_data.joblib'
VECTORIZER_PATH = '../model/final_tfidf_vectorizer_full_data.joblib'
DATA_PATH = '../../data/processed/data_for_tfidf_merged.csv'
N_BACKGROUND_SAMPLES = 100 

warnings.filterwarnings('ignore', category=UserWarning)
# ==================== KHỞI TẠO APP FLASK ====================

app = Flask(__name__)
CORS(app)
# ==================== TẢI MODEL VÀ DATA (CHỈ 1 LẦN) ====================
print(">> 🚀 Đang tải model, vectorizer và data nền...")

try:
    loaded_model = joblib.load(MODEL_PATH)
    loaded_vectorizer = joblib.load(VECTORIZER_PATH)
    feature_names = loaded_vectorizer.get_feature_names_out()

    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)
    X_full = df['text_tfidf']
    y_full = df['label']

    X_train, _, y_train, _ = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )

    X_train_tfidf = loaded_vectorizer.transform(X_train)
    background_data = shap.sample(X_train_tfidf, N_BACKGROUND_SAMPLES)

    explainer = shap.LinearExplainer(
        loaded_model,
        background_data,
        feature_perturbation="interventional"
    )

    if isinstance(explainer.expected_value, np.ndarray):
        BASE_VALUE = explainer.expected_value[0]
    else:
        BASE_VALUE = explainer.expected_value

    print(f">> ✅ Model: {loaded_model.__class__.__name__}")
    print(f">> ✅ Vectorizer: {len(feature_names)} features")
    print(f">> ✅ Explainer ready (nền {N_BACKGROUND_SAMPLES} mẫu)")
    print(f">> ✅ Base Value (Class 1): {BASE_VALUE}")
    print("\n>> 💡 API Server sẵn sàng tại http://127.0.0.1:5000")

except FileNotFoundError as e:
    print(f"LỖI FATAL: Không tìm thấy file. Hãy chắc chắn file sau có tồn tại: {e.filename}")
    print(">> API sẽ không hoạt động.")
    explainer = None # Đặt là None để API báo lỗi
except Exception as e:
    print(f"Lỗi khi tải model/data: {e}")
    explainer = None   

@app.route('/predict', methods=['POST'])
def predict():
    if explainer is None:
        return jsonify({"error": "Server-side error: Model or Explainer not loaded."}), 500
    try:
        data = request.json
        if 'text' not in data or not data['text']:
            return jsonify({"error": "No text provided"}), 400
        text_to_predict = data['text']
        instance_tfidf = loaded_vectorizer.transform([text_to_predict])
        probability_class_1 = loaded_model.predict_proba(instance_tfidf)[0][1]
        prediction_class = 1 if probability_class_1 > 0.5 else 0

        shap_values_instance = explainer.shap_values(instance_tfidf)[0]
        nonzero_indices = instance_tfidf.indices

        top_features = []
        for idx in nonzero_indices:
            feature_name = feature_names[idx]
            shap_value = shap_values_instance[idx]
            
            # Chỉ lấy các feature có ảnh hưởng (lớn hơn 0.001)
            if abs(shap_value) > 0.001:
                top_features.append({
                    "feature": feature_name,
                    "shap_value": shap_value
                })
        top_features.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        print("CLASS:", prediction_class, "PROB:", probability_class_1)
        return jsonify({
            "prediction": "REAL" if prediction_class == 1 else "FAKE",
            "probability_real": probability_class_1,
            "base_value": BASE_VALUE, 
            "top_features": top_features[:20] 
        })
        
    except Exception as e:
        print(f"Lỗi khi dự đoán: {e}")
        return jsonify({"error": f"Error during prediction: {e}"}), 500

if __name__ == '__main__':
    # Chạy Flask server
    # host='0.0.0.0' để cho phép kết nối từ bên ngoài (nếu deploy)
    # debug=True để tự động reload khi code thay đổi (chỉ dùng khi phát triển)
    app.run(host='127.0.0.1', port=5000, debug=True)
