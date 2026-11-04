import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.dataset import BrainMRIDataset, get_val_transform
from src.inference import load_ensembles
from src.utils import set_seed, get_patient_split, dice_coef_metric, iou_metric

def evaluate_classification(clf_ens, dataloader, device):
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, _, labels, _ in tqdm(dataloader, desc="Eval Classification"):
            images = images.to(device)
            labels = labels.cpu().numpy()
            
            with torch.amp.autocast(device_type=device.type, enabled=MIXED_PRECISION):
                probs = clf_ens(images).squeeze().cpu().numpy()
                
            all_probs.extend(np.atleast_1d(probs))
            all_preds.extend((np.atleast_1d(probs) > 0.5).astype(int))
            all_labels.extend(labels)
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
        
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["No Tumor", "Tumor"])
    
    return acc, f1, auc, cm, report

def evaluate_segmentation(seg_ens, dataloader, device):
    running_dice = 0.0
    running_iou = 0.0
    valid_batches = 0
    
    with torch.no_grad():
        for images, masks, labels, _ in tqdm(dataloader, desc="Eval Segmentation"):
            pos_idx = (labels > 0).view(-1)
            if not pos_idx.any():
                continue
                
            valid_batches += 1
            images = images[pos_idx].to(device)
            masks = masks[pos_idx].to(device)
            
            with torch.amp.autocast(device_type=device.type, enabled=MIXED_PRECISION):
                preds = seg_ens(images)
                
            preds = (preds > 0.5).float()
            running_dice += dice_coef_metric(preds, masks).item()
            running_iou += iou_metric(preds, masks).item()
            
    if valid_batches == 0:
        return 0, 0
        
    avg_dice = running_dice / valid_batches
    avg_iou = running_iou / valid_batches
    
    return avg_dice, avg_iou

def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Tumor", "Tumor"], yticklabels=["No Tumor", "Tumor"])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Classification Confusion Matrix')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    args = parser.parse_args()
    
    set_seed(SEED)
    _, val_patients, test_patients = get_patient_split(DATA_DIR)
    
    patients = val_patients if args.split == "val" else test_patients
    dataset = BrainMRIDataset(DATA_DIR, patients, transform=get_val_transform())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    print(f"Evaluating ensemble on {args.split} set ({len(patients)} patients)")
    clf_ens, seg_ens = load_ensembles(DEVICE)
    
    print("\n--- Classification Evaluation ---")
    acc, f1, auc, cm, report = evaluate_classification(clf_ens, dataloader, DEVICE)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC:  {auc:.4f}")
    print("\nClassification Report:")
    print(report)
    
    cm_path = os.path.join(RESULTS_DIR, f"{args.split}_confusion_matrix.png")
    plot_confusion_matrix(cm, cm_path)
    print(f"Saved confusion matrix to {cm_path}")
    
    print("\n--- Segmentation Evaluation ---")
    dice, iou = evaluate_segmentation(seg_ens, dataloader, DEVICE)
    print(f"Average Dice Score: {dice:.4f}")
    print(f"Average IoU Score:  {iou:.4f}")
