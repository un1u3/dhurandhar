"""
Model Inspector - Check what's inside your trained model checkpoint
Run this to see the exact architecture needed
"""

import torch
import sys

MODEL_PATH = 'best_xray_model.pth'

print("🔍 Inspecting Model Checkpoint...\n")
print("=" * 60)

try:
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    print(f"✅ Checkpoint loaded successfully!")
    print(f"📦 Checkpoint keys: {checkpoint.keys()}\n")
    
    # Check if it's a full checkpoint or just state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("📋 Checkpoint contains:")
        for key in checkpoint.keys():
            print(f"   - {key}")
        print()
    else:
        state_dict = checkpoint
        print("📋 This is a raw state_dict (no wrapper)")
        print()
    
    # Print all layers in the model
    print("🏗️  Model Architecture (Layers):")
    print("-" * 60)
    
    classifier_layers = []
    for key in state_dict.keys():
        if 'backbone.classifier' in key:
            classifier_layers.append(key)
    
    # Group by layer number
    layer_groups = {}
    for key in classifier_layers:
        # Extract layer number
        parts = key.split('.')
        if len(parts) >= 3 and parts[2].isdigit():
            layer_num = int(parts[2])
            if layer_num not in layer_groups:
                layer_groups[layer_num] = []
            layer_groups[layer_num].append(key)
    
    print("\n📊 Classifier Layers Found:")
    for layer_num in sorted(layer_groups.keys()):
        print(f"\n  Layer {layer_num}:")
        for key in layer_groups[layer_num]:
            shape = state_dict[key].shape
            print(f"    {key:50s} → {shape}")
    
    # Determine architecture
    print("\n" + "=" * 60)
    print("🎯 DETECTED ARCHITECTURE:")
    print("=" * 60)
    
    max_layer = max(layer_groups.keys())
    
    print("\nYour model has these layers in backbone.classifier:")
    for i in range(max_layer + 1):
        if i in layer_groups:
            sample_key = layer_groups[i][0]
            if 'weight' in sample_key:
                if 'BatchNorm' in str(type(state_dict.get(f'backbone.classifier.{i}.running_mean'))):
                    print(f"  [{i}] BatchNorm1d")
                elif len(state_dict[sample_key].shape) == 2:
                    in_feat, out_feat = state_dict[sample_key].shape[1], state_dict[sample_key].shape[0]
                    print(f"  [{i}] Linear({in_feat} → {out_feat})")
            elif 'running_mean' in sample_key:
                num_features = state_dict[sample_key].shape[0]
                print(f"  [{i}] BatchNorm1d({num_features})")
        else:
            # Infer from known patterns
            if i == 0:
                print(f"  [{i}] Dropout(0.3)")
            elif i == 3:
                print(f"  [{i}] ReLU()")
            elif i == 4:
                print(f"  [{i}] Dropout(0.3)")
    
    print("\n" + "=" * 60)
    print("💻 CORRECT PyTorch CODE:")
    print("=" * 60)
    
    # Generate the exact code needed
    print("""
class XRayClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(XRayClassifier, self).__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(""")
    
    for i in range(max_layer + 1):
        if i in layer_groups:
            sample_key = layer_groups[i][0]
            if 'running_mean' in sample_key:
                num_features = state_dict[sample_key].shape[0]
                print(f"            nn.BatchNorm1d({num_features}),")
            elif 'weight' in sample_key and len(state_dict[sample_key].shape) == 2:
                out_feat, in_feat = state_dict[sample_key].shape
                print(f"            nn.Linear({in_feat}, {out_feat}),")
        else:
            if i == 0:
                print(f"            nn.Dropout(0.3),")
            elif i == 3:
                print(f"            nn.ReLU(),")
            elif i == 4:
                print(f"            nn.Dropout(0.3),")
    
    print("""        )
    
    def forward(self, x):
        return self.backbone(x)
""")
    
    print("=" * 60)
    print("\n✅ Copy the code above and replace the model class in app.py")
    
except FileNotFoundError:
    print(f"❌ Error: {MODEL_PATH} not found!")
    print("\nMake sure you:")
    print("  1. Copied best_xray_model.pth to the same folder as this script")
    print("  2. The file name is exactly 'best_xray_model.pth'")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)