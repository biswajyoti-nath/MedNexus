import os
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.dataset import get_val_transform
from src.models.classifier import BrainClassifier
from src.models.segmentor import BrainSegmentor
from src.ensemble import ClassifierEnsemble, SegmentorEnsemble
from src.utils import load_checkpoint

def extract_bounding_box(mask):
    """
    Extracts bounding box from binary mask.
    Returns (x, y, w, h)
    """
    if mask.max() == 0:
        return None
        
    mask_np = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return (x, y, w, h)

def run_inference(image_path, classifier_ensemble, segmentor_ensemble, device):
    image_np = np.array(Image.open(image_path).convert("RGB"))
    
    transform = get_val_transform()
    augmented = transform(image=image_np)
    
    # [1, 3, 256, 256]
    img_tensor = (augmented['image'].float() / 255.0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 1. Classification
        prob = classifier_ensemble(img_tensor).item()
        
        has_tumor = prob > 0.5
        mask_np = None
        bbox = None
        
        # 2. Segmentation (only if tumor detected)
        if has_tumor:
            mask_tensor = segmentor_ensemble(img_tensor)
            mask_np = (mask_tensor.squeeze().cpu().numpy() > 0.5).astype(float)
            bbox = extract_bounding_box(mask_np)
            
    return prob, has_tumor, mask_np, bbox, image_np

def load_ensembles(device):
    classifiers = []
    for model_name in CLASSIFIERS:
        model = BrainClassifier(model_name=model_name, pretrained=False).to(device)
        path = os.path.join(CHECKPOINT_DIR, "classification", f"{model_name}_best.pth")
        if os.path.exists(path):
            load_checkpoint(model, path)
            model.eval()
            classifiers.append(model)
            
    segmentors = []
    for seg_info in SEGMENTORS:
        arch = seg_info["arch"]
        encoder = seg_info["encoder"]
        model = BrainSegmentor(architecture=arch, encoder_name=encoder).to(device)
        path = os.path.join(CHECKPOINT_DIR, "segmentation", f"{arch}_{encoder}_best.pth")
        if os.path.exists(path):
            load_checkpoint(model, path)
            model.eval()
            segmentors.append(model)
            
    # We allow running inference even if not all models are trained yet
    if not classifiers or not segmentors:
        print("Warning: Missing some trained models.")
        
    return ClassifierEnsemble(classifiers), SegmentorEnsemble(segmentors)

def visualize_prediction(image_np, has_tumor, prob, mask_np, bbox, save_path=None):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Original MRI\nTumor Prob: {prob:.4f}")
    plt.imshow(image_np)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Prediction: {'Tumor' if has_tumor else 'No Tumor'}")
    plt.imshow(image_np)
    
    if has_tumor and mask_np is not None:
        # Overlay mask (red color)
        mask_rgba = np.zeros((*mask_np.shape, 4))
        mask_rgba[mask_np == 1] = [1, 0, 0, 0.4] # Red with 0.4 alpha
        plt.imshow(mask_rgba)
        
        if bbox:
            x, y, w, h = bbox
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='yellow', linewidth=2)
            plt.gca().add_patch(rect)
            
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to MRI image slice (.tif)")
    parser.add_argument("--save", type=str, default=None, help="Path to save visualization")
    args = parser.parse_args()
    
    clf_ens, seg_ens = load_ensembles(DEVICE)
    prob, has_tumor, mask, bbox, orig_img = run_inference(args.image, clf_ens, seg_ens, DEVICE)
    visualize_prediction(orig_img, has_tumor, prob, mask, bbox, args.save)
