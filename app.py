"""
BACKEND FOR X-RAY AI MODEL - FLASK VERSION WITH GRAD-CAM
"""

from flask import Flask, request, jsonify, render_template_string
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import os
import cv2
import numpy as np
import base64
from datetime import datetime

app = Flask(__name__)

# ==============================================================================
# MODEL SETUP
# ==============================================================================

class XRayModel(nn.Module):
    """EfficientNet-B0 for X-ray classification (matched to best_xray_model.pth)"""
    
    def __init__(self, num_classes=4):
        super(XRayModel, self).__init__()
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

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==============================================================================
# GRAD-CAM IMPLEMENTATION
# ==============================================================================

class GradCAM:
    """Gradient-weighted Class Activation Mapping for model interpretability"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """Generate Grad-CAM heatmap"""
        # Forward pass
        model_output = self.model(input_image)
        
        if target_class is None:
            target_class = model_output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_score = model_output[:, target_class]
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # Apply ReLU
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, target_class

def apply_colormap_on_image(org_img, activation_map, colormap=cv2.COLORMAP_JET):
    """Apply heatmap on image"""
    # Resize activation map to match image size
    activation_map = cv2.resize(activation_map, (org_img.shape[1], org_img.shape[0]))
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * activation_map), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Superimpose heatmap on image
    superimposed_img = heatmap * 0.4 + org_img * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img

def generate_gradcam_visualization(image_tensor, original_image):
    """Generate Grad-CAM visualization"""
    # Get the last convolutional layer of EfficientNet-B0
    target_layer = model.backbone.features[-1]
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Generate CAM
    cam, predicted_class = grad_cam.generate_cam(image_tensor)
    
    # Convert original image to numpy array
    img_array = np.array(original_image.resize((224, 224)))
    
    # Apply colormap
    visualization = apply_colormap_on_image(img_array, cam)
    
    # Convert to PIL Image
    vis_image = Image.fromarray(visualization)
    
    # Convert to base64
    buffered = io.BytesIO()
    vis_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}", predicted_class

# ==============================================================================
# FLASK ROUTES
# ==============================================================================

@app.route('/')
def home():
    """Home page with professional UI"""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>X-Ray AI Diagnostic System</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body {
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
                min-height: 100vh;
                padding: 20px;
                color: #333;
            }
            
            .header {
                text-align: center;
                color: white;
                margin-bottom: 30px;
                padding: 20px;
            }
            
            .header h1 {
                font-size: 42px;
                font-weight: 700;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            
            .header p {
                font-size: 16px;
                opacity: 0.9;
                font-weight: 300;
            }
            
            .main-container {
                max-width: 1400px;
                margin: 0 auto;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
            }
            
            @media (max-width: 968px) {
                .main-container { grid-template-columns: 1fr; }
            }
            
            .card {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
                padding: 30px;
                transition: transform 0.3s ease;
            }
            
            .card:hover { transform: translateY(-5px); }
            
            .card-title {
                font-size: 24px;
                font-weight: 600;
                margin-bottom: 20px;
                color: #2c5364;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .upload-section {
                border: 3px dashed #2c5364;
                border-radius: 15px;
                padding: 40px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                position: relative;
                overflow: hidden;
            }
            
            .upload-section:hover {
                border-color: #0f2027;
                background: linear-gradient(135deg, #e0e7ef 0%, #b8c6db 100%);
                transform: scale(1.02);
            }
            
            .upload-section input { display: none; }
            
            .upload-icon {
                font-size: 64px;
                margin-bottom: 15px;
                animation: float 3s ease-in-out infinite;
            }
            
            @keyframes float {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-10px); }
            }
            
            .upload-text {
                font-size: 18px;
                font-weight: 600;
                color: #2c5364;
                margin-bottom: 8px;
            }
            
            .upload-subtext {
                font-size: 14px;
                color: #666;
            }
            
            .preview-container {
                display: none;
                margin-top: 20px;
            }
            
            .image-wrapper {
                position: relative;
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                margin-bottom: 20px;
            }
            
            .preview-image {
                width: 100%;
                height: auto;
                display: block;
                max-height: 400px;
                object-fit: contain;
                background: #000;
            }
            
            .btn-group {
                display: flex;
                gap: 15px;
                margin-top: 20px;
            }
            
            .btn {
                flex: 1;
                padding: 14px 24px;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .btn-primary {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }
            
            .btn-primary:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
            }
            
            .btn-secondary {
                background: #e0e7ef;
                color: #2c5364;
            }
            
            .btn-secondary:hover {
                background: #d0d7df;
            }
            
            .btn:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 30px;
            }
            
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto 15px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .results-panel {
                display: none;
            }
            
            .diagnosis-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 15px;
                margin-bottom: 25px;
                text-align: center;
            }
            
            .diagnosis-title {
                font-size: 32px;
                font-weight: 700;
                margin-bottom: 10px;
            }
            
            .diagnosis-confidence {
                font-size: 20px;
                opacity: 0.95;
            }
            
            .gradcam-section {
                margin: 25px 0;
            }
            
            .gradcam-title {
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 15px;
                color: #2c5364;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .gradcam-image {
                width: 100%;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            
            .predictions-list {
                background: #f8f9fa;
                border-radius: 15px;
                padding: 20px;
            }
            
            .prediction-item {
                display: flex;
                align-items: center;
                padding: 12px 0;
                border-bottom: 1px solid #e0e0e0;
            }
            
            .prediction-item:last-child { border-bottom: none; }
            
            .prediction-label {
                font-weight: 600;
                min-width: 140px;
                color: #333;
            }
            
            .prediction-bar-container {
                flex: 1;
                height: 12px;
                background: #e0e0e0;
                border-radius: 6px;
                overflow: hidden;
                margin: 0 15px;
                position: relative;
            }
            
            .prediction-bar {
                height: 100%;
                background: linear-gradient(90deg, #667eea, #764ba2);
                border-radius: 6px;
                transition: width 0.6s ease;
                box-shadow: 0 2px 4px rgba(102, 126, 234, 0.4);
            }
            
            .prediction-value {
                font-weight: 600;
                color: #667eea;
                min-width: 55px;
                text-align: right;
            }
            
            .alert {
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px 20px;
                border-radius: 10px;
                margin-top: 20px;
                display: none;
            }
            
            .alert-error {
                background: #f8d7da;
                border-left-color: #dc3545;
            }
            
            .alert-title {
                font-weight: 600;
                margin-bottom: 5px;
            }
            
            .badge {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
                background: rgba(255,255,255,0.2);
                margin-left: 10px;
            }
            
            .info-box {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                border-radius: 15px;
                padding: 20px;
                margin-top: 20px;
            }
            
            .info-title {
                font-weight: 600;
                color: #2c5364;
                margin-bottom: 10px;
                font-size: 16px;
            }
            
            .info-text {
                font-size: 14px;
                color: #555;
                line-height: 1.6;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🫁 X-Ray AI Diagnostic System</h1>
            <p>Advanced deep learning model for chest X-ray analysis with explainable AI visualization</p>
        </div>
        
        <div class="main-container">
            <!-- Left Panel: Upload Section -->
            <div class="card">
                <div class="card-title">
                    📤 Upload X-Ray Image
                </div>
                
                <div class="upload-section" id="uploadBox" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">🩻</div>
                    <div class="upload-text">Click to Upload Chest X-Ray</div>
                    <div class="upload-subtext">Supports JPG, PNG • Maximum 10MB</div>
                    <input type="file" id="fileInput" accept="image/*" onchange="handleFileSelect(event)">
                </div>
                
                <div class="preview-container" id="previewContainer">
                    <div class="image-wrapper">
                        <img id="previewImage" class="preview-image" alt="X-Ray Preview">
                    </div>
                    
                    <div class="btn-group">
                        <button class="btn btn-secondary" onclick="document.getElementById('fileInput').click()">
                            Change Image
                        </button>
                        <button class="btn btn-primary" id="analyzeBtn" onclick="analyzeImage()">
                            🔬 Analyze X-Ray
                        </button>
                    </div>
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p style="color: #2c5364; font-weight: 600;">Analyzing X-Ray Image...</p>
                    <p style="color: #666; font-size: 14px; margin-top: 10px;">Generating diagnostic report and heatmap</p>
                </div>
                
                <div class="alert alert-error" id="errorAlert">
                    <div class="alert-title">⚠️ Error</div>
                    <div id="errorMessage"></div>
                </div>
                
                <div class="info-box">
                    <div class="info-title">ℹ️ About This System</div>
                    <div class="info-text">
                        This AI system uses DenseNet-121 architecture trained on chest X-ray images. 
                        The Grad-CAM visualization highlights regions that influenced the diagnosis, 
                        helping medical professionals understand the AI's decision-making process.
                    </div>
                </div>
            </div>
            
            <!-- Right Panel: Results Section -->
            <div class="card">
                <div class="card-title">
                    📊 Diagnostic Results
                    <span class="badge" id="statusBadge" style="display: none;">ANALYZED</span>
                </div>
                
                <div class="results-panel" id="resultsPanel">
                    <div class="diagnosis-header">
                        <div class="diagnosis-title" id="diagnosisTitle">Normal</div>
                        <div class="diagnosis-confidence" id="diagnosisConfidence">Confidence: 95%</div>
                    </div>
                    
                    <div class="gradcam-section">
                        <div class="gradcam-title">
                            🔥 Grad-CAM Heatmap Visualization
                        </div>
                        <img id="gradcamImage" class="gradcam-image" alt="Grad-CAM Heatmap">
                        <div class="info-text" style="margin-top: 15px; font-size: 13px; color: #666;">
                            Red/yellow regions indicate areas with highest influence on the diagnosis. 
                            This helps identify potential infection sites or abnormalities.
                        </div>
                    </div>
                    
                    <div class="predictions-list" style="margin-top: 25px;">
                        <div style="font-weight: 600; margin-bottom: 15px; color: #2c5364; font-size: 16px;">
                            📈 All Class Probabilities
                        </div>
                        <div id="predictionsList"></div>
                    </div>
                </div>
                
                <div id="placeholderText" style="text-align: center; padding: 60px 20px; color: #999;">
                    <div style="font-size: 72px; margin-bottom: 20px;">🩺</div>
                    <p style="font-size: 18px; font-weight: 500; color: #666;">Upload an X-ray image to begin analysis</p>
                    <p style="font-size: 14px; margin-top: 10px;">Results and visualizations will appear here</p>
                </div>
            </div>
        </div>
        
        <script>
            let selectedFile = null;
            
            function handleFileSelect(event) {
                const file = event.target.files[0];
                if (!file) return;
                
                if (file.size > 10 * 1024 * 1024) {
                    showError('File size too large. Please upload an image under 10MB.');
                    return;
                }
                
                selectedFile = file;
                const reader = new FileReader();
                reader.onload = (e) => {
                    document.getElementById('previewImage').src = e.target.result;
                    document.getElementById('previewContainer').style.display = 'block';
                    document.getElementById('uploadBox').style.display = 'none';
                    document.getElementById('errorAlert').style.display = 'none';
                    document.getElementById('resultsPanel').style.display = 'none';
                    document.getElementById('placeholderText').style.display = 'block';
                    document.getElementById('statusBadge').style.display = 'none';
                };
                reader.readAsDataURL(file);
            }
            
            async function analyzeImage() {
                if (!selectedFile) return;
                
                const formData = new FormData();
                formData.append('file', selectedFile);
                
                // Show loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('resultsPanel').style.display = 'none';
                document.getElementById('placeholderText').style.display = 'none';
                document.getElementById('errorAlert').style.display = 'none';
                document.getElementById('analyzeBtn').disabled = true;
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        showError(data.error);
                    } else {
                        displayResults(data);
                    }
                } catch (error) {
                    showError('Network error: ' + error.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('analyzeBtn').disabled = false;
                }
            }
            
            function displayResults(data) {
                // Update diagnosis
                document.getElementById('diagnosisTitle').textContent = data.disease;
                document.getElementById('diagnosisConfidence').textContent = 
                    `Confidence: ${data.confidence_percent}`;
                
                // Update Grad-CAM image
                document.getElementById('gradcamImage').src = data.gradcam_image;
                
                // Update predictions list
                const predictionsHtml = Object.entries(data.all_predictions)
                    .sort(([,a], [,b]) => b - a)
                    .map(([disease, prob]) => `
                        <div class="prediction-item">
                            <div class="prediction-label">${disease}</div>
                            <div class="prediction-bar-container">
                                <div class="prediction-bar" style="width: ${prob * 100}%"></div>
                            </div>
                            <div class="prediction-value">${(prob * 100).toFixed(1)}%</div>
                        </div>
                    `)
                    .join('');
                
                document.getElementById('predictionsList').innerHTML = predictionsHtml;
                
                // Show results
                document.getElementById('resultsPanel').style.display = 'block';
                document.getElementById('placeholderText').style.display = 'none';
                document.getElementById('statusBadge').style.display = 'inline-block';
                
                // Animate bars
                setTimeout(() => {
                    document.querySelectorAll('.prediction-bar').forEach(bar => {
                        bar.style.width = bar.style.width;
                    });
                }, 100);
            }
            
            function showError(message) {
                document.getElementById('errorMessage').textContent = message;
                document.getElementById('errorAlert').style.display = 'block';
                document.getElementById('loading').style.display = 'none';
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
        'gradcam_enabled': True,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict disease from uploaded image with Grad-CAM visualization"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded on server'}), 500
        
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image and convert to RGB
        image_bytes = file.read()
        original_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess
        image_tensor = transform(original_image).unsqueeze(0).to(device)
        
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
        
        # Generate Grad-CAM visualization
        gradcam_image, _ = generate_gradcam_visualization(
            image_tensor.requires_grad_(True), 
            original_image
        )
        
        return jsonify({
            'disease': disease_name,
            'confidence': round(confidence_score, 4),
            'confidence_percent': f'{int(confidence_score * 100)}%',
            'all_predictions': all_predictions,
            'gradcam_image': gradcam_image,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        import traceback
        print(f"Error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Flask server starting...")
    print("📍 Visit http://localhost:5000")
    print("🔥 Grad-CAM visualization enabled")
    app.run(debug=True, host='0.0.0.0', port=5000)