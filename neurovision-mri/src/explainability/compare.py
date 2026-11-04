import os
import sys
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import *
from src.dataset import get_val_transform
from src.inference import load_ensembles
from src.explainability.gradcam import GradCAM, show_cam_on_image
from src.explainability.segmentation_xai import generate_agreement_map, visualize_agreement_map, mc_dropout_uncertainty, visualize_uncertainty

def generate_full_explanation(image_path, clf_ens, seg_ens, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(image_path).replace(".tif", "")
    
    image_np = np.array(Image.open(image_path).convert("RGB"))
    transform = get_val_transform()
    augmented = transform(image=image_np)
    img_tensor = (augmented['image'].float() / 255.0).unsqueeze(0).to(device)
    
    # 1. Grad-CAM for Classifiers
    # We will pick the first model (e.g., resnet34) and hook its last convolutional layer
    if len(clf_ens.models) > 0:
        model = clf_ens.models[0]
        # For timm resnet, the last conv layer is typically model.backbone.layer4[-1].conv2 or similar
        # We'll use a generic approach if possible, or target layer4
        try:
            target_layer = model.backbone.layer4[-1]
            grad_cam = GradCAM(model, target_layer)
            cam = grad_cam(img_tensor)
            
            plt.figure()
            show_cam_on_image(image_np / 255.0, cam, os.path.join(save_dir, f"{base_name}_gradcam.png"))
        except Exception as e:
            print(f"GradCAM failed: {e}")
            
    # 2. Agreement Map for Segmentors
    if len(seg_ens.models) > 0:
        masks = []
        with torch.no_grad():
            for model in seg_ens.models:
                out = model(img_tensor)
                mask = (torch.sigmoid(out).squeeze().cpu().numpy() > 0.5).astype(float)
                masks.append(mask)
                
        agreement = generate_agreement_map(masks)
        visualize_agreement_map(image_np, agreement, len(masks), os.path.join(save_dir, f"{base_name}_agreement.png"))
        
    # 3. MC Dropout Uncertainty
    if len(seg_ens.models) > 0:
        # Pick the first segmentor
        model = seg_ens.models[0]
        mean_pred, variance = mc_dropout_uncertainty(model, img_tensor, num_samples=10)
        visualize_uncertainty(image_np, variance, os.path.join(save_dir, f"{base_name}_uncertainty.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to MRI image slice (.tif)")
    parser.add_argument("--save_dir", type=str, default="results/xai", help="Directory to save explanations")
    args = parser.parse_args()
    
    clf_ens, seg_ens = load_ensembles(DEVICE)
    generate_full_explanation(args.image, clf_ens, seg_ens, DEVICE, args.save_dir)
    print(f"XAI generated in {args.save_dir}")
