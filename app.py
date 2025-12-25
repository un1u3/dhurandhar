"""
BACKEND FOR X-RAY AI MODEL - FLASK VERSION
"""

from flask import Flask, request, jsonify, render_template_string
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import os
from datetime import datetime

app = Flask(__name__)

# ==============================================================================
# MODEL SETUP
# ==============================================================================

class XRayModel(nn.Module):
    """EfficientNet-B0 for X-ray classification (matched to best_xray_model.pth)"""
    
    def __init__(self, num_classes=4):
        super(XRayModel, self).__init__()
        # The checkpoint keys show this is EfficientNet-B0 wrapped in a 'backbone' attribute
        try:
            base_model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1')
        except:
            base_model = models.efficientnet_b0(pretrained=True)
            
        base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        self.backbone = base_model
    
    def forward(self, x):
        return self.backbone(x)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
MODEL_PATH = 'best_xray_model.pth'
model = XRayModel(num_classes=4).to(device)

model_loaded = False
if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        model_loaded = True
        print(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"❌ Model file not found: {MODEL_PATH}")

# Class names
CLASS_NAMES = ['Normal', 'Pneumonia', 'TB/Consolidation', 'Infiltration']

# Image preprocessing - Updated to RGB and standard ImageNet normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==============================================================================
# FLASK ROUTES
# ==============================================================================

@app.route('/')
def home():
    """Home page with UI"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>X-Ray AI Diagnosis</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                padding: 40px;
                max-width: 600px;
                width: 100%;
            }
            h1 { color: #333; margin-bottom: 10px; text-align: center; font-size: 32px; }
            .subtitle { color: #666; text-align: center; margin-bottom: 30px; font-size: 14px; }
            .upload-box {
                border: 3px dashed #667eea;
                border-radius: 15px;
                padding: 40px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
                background: #f8f9ff;
            }
            .upload-box:hover { border-color: #764ba2; background: #f0f2ff; }
            .upload-box input { display: none; }
            .upload-icon { font-size: 48px; margin-bottom: 10px; }
            .upload-text { color: #333; font-weight: 600; margin-bottom: 5px; }
            .upload-subtext { color: #999; font-size: 12px; }
            #preview { margin-top: 20px; display: none; text-align: center; }
            #preview img { max-width: 100%; max-height: 300px; border-radius: 10px; margin-bottom: 15px; }
            #change-btn { background: #ddd; color: #666; border: none; padding: 8px 16px; border-radius: 5px; cursor: pointer; font-size: 12px; margin-bottom: 15px; }
            #analyze-btn {
                width: 100%;
                padding: 14px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-top: 15px;
            }
            #analyze-btn:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3); }
            #analyze-btn:disabled { background: #ccc; cursor: not-allowed; }
            .loading { display: none; text-align: center; margin-top: 20px; }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            .result {
                display: none;
                margin-top: 30px;
                padding: 20px;
                background: #f8f9ff;
                border-radius: 15px;
                border-left: 4px solid #667eea;
            }
            .result-title { font-size: 24px; font-weight: 700; margin-bottom: 10px; }
            .confidence { font-size: 18px; color: #667eea; font-weight: 600; margin-bottom: 15px; }
            .prediction-item { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #eee; }
            .prediction-bar { height: 8px; background: #eee; border-radius: 5px; flex-grow: 1; margin: 0 10px; overflow: hidden; }
            .prediction-bar-fill { height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 5px; }
            .error { display: none; background: #fee; color: #c33; padding: 15px; border-radius: 10px; border-left: 4px solid #c33; margin-top: 15px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🫁 X-Ray AI Diagnosis</h1>
            <p class="subtitle">Upload a chest X-ray image for instant diagnosis</p>
            <div class="upload-box" onclick="document.getElementById('imageInput').click()">
                <div class="upload-icon">📤</div>
                <div class="upload-text">Click to upload X-ray image</div>
                <div class="upload-subtext">or drag and drop (JPG, PNG)</div>
                <input type="file" id="imageInput" accept="image/*" onchange="handleImageSelect(event)">
            </div>
            <div id="preview">
                <img id="previewImg" src="" alt="Preview">
                <button id="change-btn" onclick="document.getElementById('imageInput').click()">Change Image</button>
                <button id="analyze-btn" onclick="analyzeImage()">Analyze X-Ray</button>
            </div>
            <div class="loading">
                <div class="spinner"></div>
                <p>Analyzing image...</p>
            </div>
            <div class="error" id="error"></div>
            <div class="result" id="result">
                <div class="result-title" id="resultDisease"></div>
                <div class="confidence" id="resultConfidence"></div>
                <div class="predictions" id="predictions"></div>
            </div>
        </div>
        <script>
            let selectedFile = null;
            function handleImageSelect(event) {
                const file = event.target.files[0];
                if (file) {
                    selectedFile = file;
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        document.getElementById('previewImg').src = e.target.result;
                        document.getElementById('preview').style.display = 'block';
                        document.getElementById('result').style.display = 'none';
                        document.getElementById('error').style.display = 'none';
                    };
                    reader.readAsDataURL(file);
                }
            }
            function analyzeImage() {
                if (!selectedFile) return;
                const formData = new FormData();
                formData.append('file', selectedFile);
                document.querySelector('.loading').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                document.getElementById('error').style.display = 'none';
                fetch('/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    document.querySelector('.loading').style.display = 'none';
                    if (data.error) {
                        document.getElementById('error').textContent = '❌ ' + data.error;
                        document.getElementById('error').style.display = 'block';
                    } else {
                        showResults(data);
                    }
                })
                .catch(error => {
                    document.querySelector('.loading').style.display = 'none';
                    document.getElementById('error').textContent = '❌ Error: ' + error;
                    document.getElementById('error').style.display = 'block';
                });
            }
            function showResults(data) {
                document.getElementById('resultDisease').textContent = data.disease;
                document.getElementById('resultConfidence').textContent = `Confidence: ${data.confidence_percent}`;
                const predictionsHtml = Object.entries(data.all_predictions)
                    .sort(([,a], [,b]) => b - a)
                    .map(([disease, conf]) => `
                        <div class="prediction-item">
                            <span>${disease}</span>
                            <div class="prediction-bar">
                                <div class="prediction-bar-fill" style="width: ${conf * 100}%"></div>
                            </div>
                            <span style="min-width: 50px; text-align: right;">${(conf * 100).toFixed(0)}%</span>
                        </div>
                    `)
                    .join('');
                document.getElementById('predictions').innerHTML = predictionsHtml;
                document.getElementById('result').style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'model_loaded': model_loaded,
        'classes': CLASS_NAMES,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict disease from uploaded image"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded on server'}), 500
        
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image and convert to RGB
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Preprocess
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        predicted_idx = predicted_class.item()
        confidence_score = confidence.item()
        disease_name = CLASS_NAMES[predicted_idx]
        
        all_predictions = {
            CLASS_NAMES[i]: float(probabilities[0][i].item())
            for i in range(len(CLASS_NAMES))
        }
        
        return jsonify({
            'disease': disease_name,
            'confidence': round(confidence_score, 4),
            'confidence_percent': f'{int(confidence_score * 100)}%',
            'all_predictions': all_predictions,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Flask server starting...")
    print("📍 Visit http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)