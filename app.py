"""
Skin Cancer Detection Web App
Flask backend that serves the ViT model predictions
"""

import os
import json
import io
import base64

import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from transformers import ViTImageProcessor, ViTForImageClassification

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'final_model')

# Confidence thresholds
# If the model's top prediction is below these levels it likely isn't a skin lesion
THRESHOLD_REJECT = 0.30   # < 30% → "no lesion detected"
THRESHOLD_WARN   = 0.45   # 30-45% → "low confidence"

# Load model once at startup
print(f"Loading model from {MODEL_PATH}...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
model = ViTForImageClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

with open(os.path.join(MODEL_PATH, 'config.json'), 'r') as f:
    _config = json.load(f)
    ID2LABEL = {int(k): v for k, v in _config['id2label'].items()}

# Classes considered higher-risk
HIGH_RISK   = {'malignant', 'basal cell carcinoma', 'squamous cell carcinoma', 'actinic keratosis'}
# If the retrained model explicitly outputs 'normal', treat as no-lesion
NORMAL_CLASS = 'normal'

HAS_NORMAL_CLASS = NORMAL_CLASS in ID2LABEL.values()
print(f"Model ready. Classes: {list(ID2LABEL.values())}")
print(f"Normal class present in model: {HAS_NORMAL_CLASS}")


def prediction_entropy(probs):
    """Normalised Shannon entropy — 0 = certain, 1 = totally uncertain."""
    n = len(probs)
    eps = 1e-9
    raw = -np.sum(probs * np.log(probs + eps))
    return float(raw / np.log(n))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    allowed = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in allowed:
        return jsonify({'error': f'File type .{ext} not supported. Use JPG, PNG, etc.'}), 400

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Run inference
        inputs = processor(images=image, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        predicted_idx = int(np.argmax(probs))
        predicted_label = ID2LABEL[predicted_idx]
        confidence = float(probs[predicted_idx])
        entropy = prediction_entropy(probs)

        # Determine result type
        # Priority 1: retrained model explicitly predicts 'normal'
        # Priority 2: confidence/entropy thresholds (fallback for old 10-class model)
        if predicted_label == NORMAL_CLASS:
            result_type = 'no_lesion'
        elif confidence < THRESHOLD_REJECT or entropy > 0.85:
            result_type = 'no_lesion'
        elif confidence < THRESHOLD_WARN:
            result_type = 'low_confidence'
        else:
            result_type = 'detected'

        top5_idx = np.argsort(probs)[-5:][::-1]
        top_predictions = [
            {
                'label': ID2LABEL[int(i)],
                'probability': float(probs[i]),
                'high_risk': ID2LABEL[int(i)] in HIGH_RISK,
            }
            for i in top5_idx
        ]

        # Encode image for preview in response
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        img_mime = f'image/{ext if ext != "jpg" else "jpeg"}'

        return jsonify({
            'result_type': result_type,       # 'detected' | 'low_confidence' | 'no_lesion'
            'prediction': predicted_label,
            'confidence': confidence,
            'entropy': entropy,
            'high_risk': predicted_label in HIGH_RISK,
            'top_predictions': top_predictions,
            'image_data': f'data:{img_mime};base64,{img_b64}',
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
