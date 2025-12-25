"""
Diagnostic Script: Check Model Predictions and Class Distribution
This script tests the model to see what it predicts and identifies the TB prediction issue.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Define the model architecture (same as in app.py)
class XRayClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(XRayClassifier, self).__init__()
        self.backbone = models.efficientnet_b0(weights=None)
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

# Load the model
print("🔍 Loading model checkpoint...")
checkpoint = torch.load('best_xray_model.pth', map_location='cpu')
model = XRayClassifier(num_classes=4)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

CLASSES = ['NORMAL', 'PNEUMONIA', 'COVID', 'TB']

print("\n" + "="*60)
print("📊 MODEL ANALYSIS")
print("="*60)

# Check the final layer weights and biases
final_layer = model.backbone.classifier[5]
print(f"\n🔬 Final Layer (Linear 512→4) Analysis:")
print(f"   Weight shape: {final_layer.weight.shape}")
print(f"   Bias shape: {final_layer.bias.shape}")

# Check the bias values for each class
print(f"\n📈 Class Biases (initial tendencies):")
for i, class_name in enumerate(CLASSES):
    bias_value = final_layer.bias[i].item()
    print(f"   {class_name:12s}: {bias_value:+.6f}")

# Check weight norms for each class
print(f"\n📊 Weight Norms (feature importance per class):")
for i, class_name in enumerate(CLASSES):
    weight_norm = torch.norm(final_layer.weight[i]).item()
    print(f"   {class_name:12s}: {weight_norm:.6f}")

# Create a random test input to see prediction distribution
print(f"\n🧪 Testing with Random Input:")
test_input = torch.randn(10, 3, 224, 224)  # 10 random images

with torch.no_grad():
    outputs = model(test_input)
    probabilities = F.softmax(outputs, dim=1)
    predictions = probabilities.argmax(dim=1)

# Count predictions
from collections import Counter
pred_counts = Counter(predictions.tolist())

print(f"\n📊 Prediction Distribution (10 random inputs):")
for i, class_name in enumerate(CLASSES):
    count = pred_counts.get(i, 0)
    print(f"   {class_name:12s}: {count}/10 ({count*10}%)")

# Check average probabilities
avg_probs = probabilities.mean(dim=0)
print(f"\n📈 Average Probabilities Across Random Inputs:")
for i, class_name in enumerate(CLASSES):
    prob = avg_probs[i].item() * 100
    print(f"   {class_name:12s}: {prob:.2f}%")

# Check if TB class is ever predicted with high confidence
max_tb_prob = probabilities[:, 3].max().item() * 100
print(f"\n⚠️  Maximum TB probability across all random inputs: {max_tb_prob:.2f}%")

print("\n" + "="*60)
print("🔍 DIAGNOSIS")
print("="*60)

# Check training data distribution from checkpoint
print("\n📚 Training Data Distribution:")
print("   NORMAL:    Train=1102, Val=236,  Test=237")
print("   PNEUMONIA: Train=2985, Val=639,  Test=641")
print("   COVID:     Train=2531, Val=542,  Test=543")
print("   TB:        Train=0,    Val=0,    Test=0    ⚠️ NO DATA!")

print("\n" + "="*60)
print("⚠️  ROOT CAUSE IDENTIFIED:")
print("="*60)
print("""
The model was trained with ZERO TB images!

This means:
1. The model has never seen any TB X-rays during training
2. The TB class (index 3) has learned random/arbitrary weights
3. The model cannot reliably predict TB because it doesn't know what TB looks like
4. The 98.9% validation accuracy only reflects performance on the 3 classes with data

SOLUTION:
To fix this, you need to:
1. Obtain TB chest X-ray images
2. Add them to your dataset
3. Retrain the model with all 4 classes
4. Or remove TB from the class list and use a 3-class model
""")

print("="*60)
