import os
import io
import base64
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, render_template, jsonify
from torchvision import transforms
# GradCAM imports (ensure you install pytorch-grad-cam)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

# --- CUSTOM MODEL ARCHITECTURE (Must match the one used during training) ---

class XRayClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(XRayClassifier, self).__init__()
        from torchvision import models # Import models here to avoid global import issues if needed
        
        # Using EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=None)

        # Replace classifier - must match the trained model architecture exactly
        # The checkpoint has: [0] Dropout, [1] Linear(1280->512), [2] ReLU, [3] BatchNorm1d(512), [4] Dropout, [5] Linear(512->4)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),              # [0]
            nn.Linear(1280, 512),         # [1]
            nn.ReLU(),                    # [2]
            nn.BatchNorm1d(512),          # [3]
            nn.Dropout(0.3),              # [4]
            nn.Linear(512, 4)             # [5]
        )

    def forward(self, x):
        return self.backbone(x)

# GradCAM Helper Class (Must match the one used during training)
class GradCAMVisualizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        # Target the last convolutional layer of EfficientNet-B0
        target_layer = model.backbone.features[-1]
        self.cam = GradCAM(model=model, target_layers=[target_layer])

    def generate_heatmap(self, image_tensor, predicted_class):
        """Generate GradCAM heatmap for an image"""
        # image_tensor should be 1x3x224x224 for cam
        # Pass targets=None to use the highest scoring class automatically
        grayscale_cam = self.cam(input_tensor=image_tensor.unsqueeze(0),
                                  targets=None)
        return grayscale_cam[0] # Returns numpy array 224x224


# --- CONFIGURATION & SETUP ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables for model state
MODEL = None
CLASSES = None
TRANSFORM = None
GRADCAM_OBJ = None


def load_model():
    """Loads the model and configuration files."""
    global MODEL, CLASSES, TRANSFORM, GRADCAM_OBJ

    # Check if model is already loaded (redundant check, but good practice)
    if MODEL is not None:
        return

    try:
        # Define classes (from training)
        CLASSES = ['NORMAL', 'PNEUMONIA', 'COVID', 'TB']

        # Load the checkpoint
        checkpoint = torch.load('best_xray_model.pth', map_location=DEVICE)
        
        # Initialize the model with correct architecture
        MODEL = XRayClassifier(num_classes=4)
        
        # Load the state dict from checkpoint
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL.to(DEVICE)
        MODEL.eval()

        # Define the validation/test transform (must match training)
        TRANSFORM = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Initialize GradCAM
        GRADCAM_OBJ = GradCAMVisualizer(MODEL, DEVICE)

        print("✅ Model, transforms, and GradCAM loaded successfully.")
        print(f"📊 Classes: {CLASSES}")
        print(f"🎯 Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
        print(f"📈 Best validation accuracy: {checkpoint.get('val_acc', 'unknown')}")
        print(f"\n⚠️  IMPORTANT: Training data distribution:")
        print(f"   - NORMAL: 1,102 images ✅")
        print(f"   - PNEUMONIA: 2,985 images ✅")
        print(f"   - COVID: 2,531 images ✅")
        print(f"   - TB: 0 images ❌ (NOT TRAINED - predictions unreliable!)")
        print()
    except Exception as e:
        print(f"❌ Error loading model or config: {e}")
        import traceback
        traceback.print_exc()
        # Raising an exception here will stop the app from handling requests
        raise RuntimeError(f"Model loading failed: {e}")

def image_to_base64(img_array):
    """Converts a numpy image array (0-1 float) to a base64 string for HTML display."""
    # Convert numpy array to PIL Image (scale back to 0-255)
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    # Save image to an in-memory buffer
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    # Encode to base64
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def predict_xray(image_path):
    """Performs inference and generates GradCAM visualization."""
    global MODEL, CLASSES, TRANSFORM, GRADCAM_OBJ

    if MODEL is None:
        # This should ideally not happen if @app.before_request works, but is a safety check.
        raise RuntimeError("Model is not loaded. Cannot perform prediction.")

    # 1. Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    # Original image for overlay, resized to model input size (224x224)
    original_img_np = np.array(image.resize((224, 224))) / 255.0

    image_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    # 2. Predict
    with torch.no_grad():
        output = MODEL(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_idx_tensor = probabilities.max(1)
    
    predicted_idx = predicted_idx_tensor.item()
    prediction = CLASSES[predicted_idx]
    confidence_score = confidence.item() * 100
    
    # 3. Generate GradCAM heatmap
    heatmap_2d_np = GRADCAM_OBJ.generate_heatmap(image_tensor[0], predicted_idx)

    # 4. Create GradCAM Overlay Image
    cam_image_np = show_cam_on_image(original_img_np, heatmap_2d_np, use_rgb=True)
    
    # Convert images to base64 strings
    original_img_b64 = image_to_base64(original_img_np)
    cam_img_b64 = image_to_base64(cam_image_np)

    # 5. Format results
    results = {
        'prediction': prediction,
        'confidence': f"{confidence_score:.2f}",
        'original_img_b64': original_img_b64,
        'cam_img_b64': cam_img_b64,
        'all_probabilities': {CLASSES[i]: f"{p*100:.2f}" for i, p in enumerate(probabilities[0].tolist())}
    }
    
    # Add warning if TB is predicted (model was not trained on TB data)
    if prediction == 'TB':
        results['warning'] = (
            "⚠️ WARNING: This model was not trained on any TB X-ray images. "
            "TB predictions are unreliable and should not be trusted. "
            "Please consult a medical professional for accurate TB diagnosis."
        )
    
    return results

# --- FLASK ROUTES ---

# FIX for the 'AttributeError: 'Flask' object has no attribute 'before_first_request''
@app.before_request
def check_model_loading():
    """Load model only if it hasn't been loaded yet (runs before the first request)."""
    global MODEL
    if MODEL is None:
        try:
            load_model()
        except RuntimeError as e:
            # Re-raise the error to stop the request and signal a 500 server error
            raise e

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_predict():
    """Handle image upload and prediction request."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the uploaded file temporarily
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            results = predict_xray(filepath)
            # Add classes to results for frontend display
            results['classes'] = CLASSES
            return jsonify(results)
        except Exception as e:
            print(f"Prediction Error: {e}")
            return jsonify({'error': f'Prediction failed: {e}'}), 500
        finally:
            # Clean up the file
            if os.path.exists(filepath):
                os.remove(filepath)
            
if __name__ == '__main__':
    # Use 0.0.0.0 for external access (e.g., if deploying on a cloud VM)
    # Use threaded=False if running on CUDA to avoid re-initializing CUDA context
    # debug=True is useful for development but should be False for production
    app.run(host='0.0.0.0', port=8000, debug=False)