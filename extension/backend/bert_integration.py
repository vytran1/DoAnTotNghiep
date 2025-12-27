import torch
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
import logging
from datetime import datetime
import sys
import io
import json

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
MODEL_DIR = '../model/BERRT_model_result/'  # Thư mục chứa các file BERT của bạn
MAX_TEXT_LENGTH = 512  # BERT max sequence length
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings('ignore', category=UserWarning)

# ==================== KHỞI TẠO APP FLASK ====================
app = Flask(__name__)
CORS(app)

# ==================== TẢI MODEL VÀ TOKENIZER ====================
logger.info("="*70)
logger.info("Bắt đầu khởi động API Server với BERT...")
logger.info("="*70)

try:
    logger.info(f"🔧 Device: {DEVICE}")
    
    logger.info("📥 Đang tải BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    logger.info(f"✅ Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    
    logger.info("📥 Đang tải BERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()  # Set to evaluation mode
    logger.info(f"✅ Model loaded: {model.config.num_labels} classes")
    
    # Load config nếu có
    try:
        with open(f"{MODEL_DIR}/config.json", 'r') as f:
            model_config = json.load(f)
            logger.info(f"📋 Model config: {model_config.get('model_type', 'bert')}")
    except:
        logger.warning("⚠️ Could not load config.json")
    
    logger.info("="*70)
    logger.info("✨ API Server sẵn sàng tại http://127.0.0.1:5000")
    logger.info("="*70)

except FileNotFoundError as e:
    logger.error(f"❌ FATAL: Không tìm thấy file: {e}")
    logger.error("API sẽ không hoạt động.")
    model = None
    tokenizer = None
except Exception as e:
    logger.error(f"❌ Lỗi khi tải model/tokenizer: {e}", exc_info=True)
    model = None
    tokenizer = None

# ==================== HELPER FUNCTIONS ====================

def predict_with_bert(text):
    """
    Dự đoán với BERT model
    
    Returns:
        dict: {
            'prediction': int (0 or 1),
            'probability_fake': float,
            'probability_real': float,
            'confidence': float,
            'logits': list
        }
    """
    # Tokenize
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_TEXT_LENGTH,
        return_tensors="pt"
    )
    
    # Move to device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    # Get probabilities
    probs = torch.softmax(logits, dim=-1)
    probs = probs.cpu().numpy()[0]
    
    # Get prediction
    prediction = int(np.argmax(probs))
    confidence = float(np.max(probs))
    
    return {
        'prediction': prediction,
        'probability_fake': float(probs[0]),
        'probability_real': float(probs[1]),
        'confidence': confidence,
        'logits': logits.cpu().numpy()[0].tolist()
    }

def get_attention_weights(text):
    """
    Lấy attention weights từ BERT (optional - for explainability)
    """
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_TEXT_LENGTH,
        return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
    # Get last layer attention
    attention = outputs.attentions[-1]  # Last layer
    attention = attention.mean(dim=1)  # Average over heads
    attention = attention[0].cpu().numpy()  # First (only) item in batch
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Get important tokens (high attention to [CLS])
    cls_attention = attention[0, 1:]  # Attention from [CLS] to other tokens
    
    important_tokens = []
    for i, (token, att) in enumerate(zip(tokens[1:], cls_attention)):
        if token not in ['[SEP]', '[PAD]'] and att > 0.01:
            important_tokens.append({
                'token': token,
                'attention': float(att),
                'position': i
            })
    
    # Sort by attention
    important_tokens.sort(key=lambda x: x['attention'], reverse=True)
    
    return important_tokens[:20]  # Top 20 tokens

# ==================== ROUTES ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": str(DEVICE),
        "model_type": "BERT",
        "timestamp": datetime.now().isoformat()
    }
    logger.info(f"🏥 Health check: {status['status']}")
    return jsonify(status)

@app.route('/predict', methods=['POST'])
def predict():
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    logger.info("="*50)
    logger.info(f"🆕 NEW REQUEST [{request_id}]")
    
    if model is None or tokenizer is None:
        logger.error(f"[{request_id}] ❌ Model not loaded")
        return jsonify({"error": "Server-side error: Model not loaded"}), 500
    
    try:
        # Validate request
        data = request.json
        if not data or 'text' not in data or not data['text']:
            logger.warning(f"[{request_id}] ⚠️ No text provided")
            return jsonify({"error": "No text provided"}), 400
        
        text_to_predict = data['text']
        text_length = len(text_to_predict)
        
        # Log input info
        logger.info(f"[{request_id}] 📝 Text length: {text_length} chars")
        logger.info(f"[{request_id}] 👀 Preview: {text_to_predict[:100]}...")
        
        # Validate text length
        if text_length < 10:
            logger.warning(f"[{request_id}] ⚠️ Text too short: {text_length} chars")
            return jsonify({"error": "Text too short. Minimum 10 characters"}), 400
        
        # Predict with BERT
        logger.info(f"[{request_id}] 🤖 Making prediction with BERT...")
        prediction_result = predict_with_bert(text_to_predict)
        
        prediction_class = prediction_result['prediction']
        probability_real = prediction_result['probability_real']
        probability_fake = prediction_result['probability_fake']
        confidence = prediction_result['confidence']
        
        # Log prediction results
        logger.info(f"[{request_id}] 🎯 PREDICTION: {'REAL' if prediction_class == 1 else 'FAKE'}")
        logger.info(f"[{request_id}] 📊 Prob REAL: {probability_real:.4f} ({probability_real*100:.2f}%)")
        logger.info(f"[{request_id}] 📊 Prob FAKE: {probability_fake:.4f} ({probability_fake*100:.2f}%)")
        logger.info(f"[{request_id}] 💪 Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        
        # Get attention-based explanations (optional)
        logger.info(f"[{request_id}] 🔍 Getting attention weights...")
        try:
            important_tokens = get_attention_weights(text_to_predict)
            logger.info(f"[{request_id}] 🎯 Top 5 important tokens:")
            for i, token_info in enumerate(important_tokens[:5], 1):
                logger.info(f"   {i}. '{token_info['token']}': {token_info['attention']:.4f}")
        except Exception as e:
            logger.warning(f"[{request_id}] ⚠️ Could not get attention weights: {e}")
            important_tokens = []
        
        # Prepare response
        response_data = {
            "prediction": "REAL" if prediction_class == 1 else "FAKE",
            "prediction_label": int(prediction_class),
            "probability_real": probability_real,
            "probability_fake": probability_fake,
            "confidence": confidence,
            "logits": prediction_result['logits'],
            "important_tokens": important_tokens,
            "metadata": {
                "text_length": text_length,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "model_type": "BERT",
                "device": str(DEVICE)
            }
        }
        
        logger.info(f"[{request_id}] ✅ Request completed successfully")
        logger.info("="*50)
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"[{request_id}] ❌ ERROR: {str(e)}", exc_info=True)
        logger.info("="*50)
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint for multiple texts"""
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    logger.info("="*50)
    logger.info(f"🆕 BATCH REQUEST [{request_id}]")
    
    if model is None or tokenizer is None:
        return jsonify({"error": "Server-side error: Model not loaded"}), 500
    
    try:
        data = request.json
        if not data or 'texts' not in data:
            return jsonify({"error": "No texts provided"}), 400
        
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({"error": "texts must be a list"}), 400
        
        logger.info(f"[{request_id}] 📦 Processing {len(texts)} texts")
        
        results = []
        for i, text in enumerate(texts):
            try:
                pred_result = predict_with_bert(text)
                results.append({
                    "index": i,
                    "prediction": "REAL" if pred_result['prediction'] == 1 else "FAKE",
                    "probability_real": pred_result['probability_real'],
                    "confidence": pred_result['confidence']
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "error": str(e)
                })
        
        logger.info(f"[{request_id}] ✅ Batch completed: {len(results)} results")
        logger.info("="*50)
        
        return jsonify({
            "results": results,
            "metadata": {
                "total": len(texts),
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"[{request_id}] ❌ ERROR: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ==================== RUN SERVER ====================
if __name__ == '__main__':
    logger.info("\n🚀 Starting Flask development server...")
    app.run(host='127.0.0.1', port=5000, debug=True)