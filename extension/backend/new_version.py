import joblib
import shap
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse
import warnings
import logging
from datetime import datetime
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')



# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('api_logs.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==================== CONFIG ====================
MODEL_PATH = '../model/five_training/final_model_full_data.joblib'
VECTORIZER_PATH = '../model/five_training/final_tfidf_vectorizer_full_data.joblib'
DATA_PATH = '../../data/processed/data_for_tfidf_merged.csv'
TEST_DATA_PATH = '../../data/processed/test_data_tfidf_clean.csv'
TRAIN_DATA_PATH = '../../data/processed/train_data_tfidf_clean.csv'
N_BACKGROUND_SAMPLES = 100
MAX_TEXT_LENGTH = 70000

warnings.filterwarnings('ignore', category=UserWarning)

# ==================== KHỞI TẠO APP FLASK ====================
app = Flask(__name__)
CORS(app)

# ==================== TẢI MODEL VÀ DATA ====================
logger.info("="*70)
logger.info("Bắt đầu khởi động API Server...")
logger.info("="*70)

try:
    logger.info(" Đang tải model...")
    loaded_model = joblib.load(MODEL_PATH)
    logger.info(f" Model loaded: {loaded_model.__class__.__name__}")
    
    logger.info("Đang tải vectorizer...")
    loaded_vectorizer = joblib.load(VECTORIZER_PATH)
    feature_names = loaded_vectorizer.get_feature_names_out()
    logger.info(f"Vectorizer loaded: {len(feature_names)} features")

    # --- THAY ĐỔI Ở ĐÂY: Tải dữ liệu huấn luyện (CHỈ DÙNG CHO SHAP BACKGROUND) ---
    logger.info("Đang tải dataset huấn luyện...")
    # Tải dữ liệu huấn luyện
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_train.dropna(inplace=True)
    X_train = df_train['text_tfidf']
    # y_train = df_train['label'] # Không cần y_train cho SHAP background
    logger.info(f"Dataset huấn luyện loaded: {len(df_train)} samples")

    # Tải dữ liệu kiểm thử (Chỉ để log, không dùng cho SHAP)
    df_test = pd.read_csv(TEST_DATA_PATH)
    df_test.dropna(inplace=True)
    logger.info(f"Dataset kiểm thử loaded: {len(df_test)} samples")
    # ---------------------------------------------------------------------------------

    logger.info("Đang chuẩn bị dữ liệu huấn luyện (TF-IDF) cho SHAP...")
    # Chuyển đổi X_train thành TF-IDF matrix
    X_train_tfidf = loaded_vectorizer.transform(X_train)
    
    # Lấy mẫu background data từ X_train_tfidf
    background_data = shap.sample(X_train_tfidf, N_BACKGROUND_SAMPLES)
    logger.info(f"Background data: {N_BACKGROUND_SAMPLES} samples")

    logger.info(" Đang khởi tạo SHAP Explainer...")
    explainer = shap.LinearExplainer(
        loaded_model,
        background_data,
        feature_perturbation="interventional"
    )

    if isinstance(explainer.expected_value, np.ndarray):
        BASE_VALUE = explainer.expected_value[0]
    else:
        BASE_VALUE = explainer.expected_value

    logger.info(f" Explainer ready - Base Value: {BASE_VALUE:.4f}")
    logger.info("="*70)
    logger.info("API Server sẵn sàng tại http://127.0.0.1:5000")
    logger.info("="*70)

except FileNotFoundError as e:
    logger.error(f"FATAL: Không tìm thấy file: {e.filename}")
    logger.error("API sẽ không hoạt động.")
    explainer = None
except Exception as e:
    logger.error(f"Lỗi khi tải model/data: {e}", exc_info=True)
    explainer = None

# ==================== ROUTES ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy" if explainer is not None else "unhealthy",
        "model_loaded": explainer is not None,
        "features_count": len(feature_names) if explainer else 0,
        "timestamp": datetime.now().isoformat()
    }
    logger.info(f"Health check: {status['status']}")
    return jsonify(status)

@app.route('/predict', methods=['POST'])
def predict():
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    logger.info("="*50)
    logger.info(f"NEW REQUEST [{request_id}]")
    
    if explainer is None:
        logger.error(f"[{request_id}] Model not loaded")
        return jsonify({"error": "Server-side error: Model not loaded"}), 500
    
    try:
        # Validate request
        data = request.json
        if not data or 'text' not in data or not data['text']:
            logger.warning(f"[{request_id}] No text provided")
            return jsonify({"error": "No text provided"}), 400
        
        text_to_predict = data['text']
        text_length = len(text_to_predict)
        
        # Log input info
        logger.info(f"[{request_id}] Text length: {text_length} chars")
        logger.info(f"[{request_id}] Preview: {text_to_predict[:100]}...")
        
        # Validate text length
        if text_length > MAX_TEXT_LENGTH:
            logger.warning(f"[{request_id}] Text too long: {text_length} > {MAX_TEXT_LENGTH}")
            return jsonify({"error": f"Text too long. Max {MAX_TEXT_LENGTH} characters"}), 400
        
        if text_length < 10:
            logger.warning(f"[{request_id}] Text too short: {text_length} chars")
            return jsonify({"error": "Text too short. Minimum 10 characters"}), 400
        
        # Transform and predict
        logger.info(f"[{request_id}] Transforming text...")
        instance_tfidf = loaded_vectorizer.transform([text_to_predict])
        
        logger.info(f"[{request_id}] Making prediction...")
        probability_class_1 = loaded_model.predict_proba(instance_tfidf)[0][1]
        prediction_class = 1 if probability_class_1 > 0.5 else 0
        confidence = max(probability_class_1, 1 - probability_class_1)
        
        # Log prediction results
        logger.info(f"[{request_id}] PREDICTION: {'REAL' if prediction_class == 1 else 'FAKE'}")
        logger.info(f"[{request_id}] Prob REAL: {probability_class_1:.4f} ({probability_class_1*100:.2f}%)")
        logger.info(f"[{request_id}] Prob FAKE: {1-probability_class_1:.4f} ({(1-probability_class_1)*100:.2f}%)")
        logger.info(f"[{request_id}] Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        
        # Calculate SHAP
        logger.info(f"[{request_id}] Calculating SHAP values...")
        shap_values_instance = explainer.shap_values(instance_tfidf)[0]
        nonzero_indices = instance_tfidf.indices
        
        logger.info(f"[{request_id}] Found {len(nonzero_indices)} non-zero features")
        
        # Extract top features
        top_features = []
        for idx in nonzero_indices:
            feature_name = feature_names[idx]
            shap_value = shap_values_instance[idx]
            
            if abs(shap_value) > 0.001:
                top_features.append({
                    "feature": feature_name,
                    "shap_value": float(shap_value)
                })
        
        top_features.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        # Log top features
        logger.info(f"[{request_id}] Top 5 features:")
        for i, feat in enumerate(top_features[:5], 1):
            direction = "→ REAL" if feat['shap_value'] > 0 else "→ FAKE"
            logger.info(f"   {i}. '{feat['feature']}': {feat['shap_value']:.4f} {direction}")
        
        # Prepare response
        response_data = {
            "prediction": "REAL" if prediction_class == 1 else "FAKE",
            "probability_real": float(probability_class_1),
            "probability_fake": float(1 - probability_class_1),
            "confidence": float(confidence),
            "base_value": float(BASE_VALUE),
            "top_features": top_features[:20],
            "metadata": {
                "text_length": text_length,
                "num_features": len(nonzero_indices),
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        logger.info(f"[{request_id}] Request completed successfully")
        logger.info("="*50)
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"[{request_id}] ERROR: {str(e)}", exc_info=True)
        logger.info("="*50)
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

# ==================== RUN SERVER ====================
if __name__ == '__main__':
    logger.info("\nStarting Flask development server...")
    app.run(host='127.0.0.1', port=5000, debug=True)


# import joblib
# import shap
# import numpy as np
# import pandas as pd
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import warnings
# import logging
# from datetime import datetime
# import sys
# import io

# # Đảm bảo output hỗ trợ tiếng Việt/ký tự đặc biệt
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# # ==================== LOGGING SETUP ====================
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s [%(levelname)s] %(message)s',
#     handlers=[
#         logging.FileHandler('api_logs.log', encoding='utf-8'),
#         logging.StreamHandler(sys.stdout)
#     ]
# )
# logger = logging.getLogger(__name__)

# # ==================== CONFIG ====================
# MODEL_PATH = '../model/five_training/final_model_full_data.joblib'
# VECTORIZER_PATH = '../model/five_training/final_tfidf_vectorizer_full_data.joblib'
# TRAIN_DATA_PATH = '../../data/processed/train_data_tfidf_clean.csv'
# N_BACKGROUND_SAMPLES = 100
# MAX_TEXT_LENGTH = 70000

# warnings.filterwarnings('ignore', category=UserWarning)

# # ==================== KHỞI TẠO APP FLASK ====================
# app = Flask(__name__)
# CORS(app)

# # ==================== TẢI MODEL VÀ DATA ====================
# logger.info("="*70)
# logger.info("Bắt đầu khởi động API Server (ML - TF-IDF)...")
# logger.info("Logic Nhãn: 0 = REAL, 1 = FAKE")
# logger.info("="*70)

# try:
#     logger.info("🔹 Đang tải model...")
#     loaded_model = joblib.load(MODEL_PATH)
    
#     logger.info("🔹 Đang tải vectorizer...")
#     loaded_vectorizer = joblib.load(VECTORIZER_PATH)
#     feature_names = loaded_vectorizer.get_feature_names_out()

#     logger.info("🔹 Đang chuẩn bị SHAP background từ train data...")
#     df_train = pd.read_csv(TRAIN_DATA_PATH)
#     df_train.dropna(inplace=True)
#     X_train_tfidf = loaded_vectorizer.transform(df_train['text_tfidf'])
#     background_data = shap.sample(X_train_tfidf, N_BACKGROUND_SAMPLES)

    
#     explainer = shap.LinearExplainer(
#         loaded_model,
#         background_data,
#         feature_perturbation="interventional"
#     )

    
#     if isinstance(explainer.expected_value, (np.ndarray, list)):
#         BASE_VALUE = explainer.expected_value[1] # Lấy base value của class 1
#     else:
#         BASE_VALUE = explainer.expected_value

#     logger.info(f"✅ Hệ thống sẵn sàng - Base Value (FAKE): {BASE_VALUE:.4f}")

# except Exception as e:
#     logger.error(f"❌ Lỗi khởi động: {e}", exc_info=True)
#     explainer = None

# # ==================== ROUTES ====================

# @app.route('/predict', methods=['POST'])
# def predict():
#     request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#     logger.info(f"NEW REQUEST [{request_id}]")
    
#     if explainer is None:
#         return jsonify({"error": "Model not loaded"}), 500
    
#     try:
#         data = request.json
#         text_to_predict = data.get('text', '')
        
#         if not text_to_predict or len(text_to_predict) < 10:
#             return jsonify({"error": "Text invalid hoặc quá ngắn"}), 400

        
#         instance_tfidf = loaded_vectorizer.transform([text_to_predict])
#         probs = loaded_model.predict_proba(instance_tfidf)[0]
        
        
#         prob_real = float(probs[0])
#         prob_fake = float(probs[1])
        
#         prediction_label = "FAKE" if prob_fake > 0.5 else "REAL"
#         confidence = max(prob_real, prob_fake)

        
#         shap_values_all = explainer.shap_values(instance_tfidf)
        
        
#         if isinstance(shap_values_all, list):
#             shap_values_instance = shap_values_all[1]
#         else:
#             shap_values_instance = shap_values_all
            
#         nonzero_indices = instance_tfidf.indices
#         top_features = []
        
#         for idx in nonzero_indices:
#             feature_name = feature_names[idx]
#             val = shap_values_instance[idx]
#             if abs(val) > 0.001:
#                 top_features.append({
#                     "feature": feature_name,
#                     "shap_value": float(val) 
#                 })
        
        
#         top_features.sort(key=lambda x: abs(x['shap_value']), reverse=True)

#         # 3. Response
#         response_data = {
#             "prediction": prediction_label,
#             "probability_real": prob_real,
#             "probability_fake": prob_fake,
#             "confidence": confidence,
#             "base_value": float(BASE_VALUE),
#             "top_features": top_features[:20],
#             "metadata": {
#                 "request_id": request_id,
#                 "num_features_detected": len(nonzero_indices)
#             }
#         }
        
#         logger.info(f"[{request_id}] Result: {prediction_label} | Fake: {prob_fake:.2f}")
#         return jsonify(response_data)
        
#     except Exception as e:
#         logger.error(f"[{request_id}] ERROR: {str(e)}", exc_info=True)
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='127.0.0.1', port=5000, debug=True)