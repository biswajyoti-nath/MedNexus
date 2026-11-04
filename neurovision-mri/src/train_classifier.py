import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.utils import set_seed, get_patient_split, save_checkpoint
from src.dataset import BrainMRIDataset, get_train_transform, get_val_transform
from src.models.classifier import BrainClassifier

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for images, masks, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type=device.type, enabled=MIXED_PRECISION):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend((preds > 0.5).astype(int))
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})
        
    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_labels, all_preds)
    return epoch_loss, epoch_f1

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, masks, labels, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            with torch.amp.autocast(device_type=device.type, enabled=MIXED_PRECISION):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            running_loss += loss.item()
            
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend((probs > 0.5).astype(int))
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_labels, all_preds)
    epoch_acc = accuracy_score(all_labels, all_preds)
    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        epoch_auc = 0.5
        
    return epoch_loss, epoch_f1, epoch_acc, epoch_auc

def main(args):
    set_seed(SEED)
    
    train_patients, val_patients, _ = get_patient_split(DATA_DIR)
    print(f"Training on {len(train_patients)} patients, Validation on {len(val_patients)} patients")
    
    train_dataset = BrainMRIDataset(DATA_DIR, train_patients, transform=get_train_transform())
    val_dataset = BrainMRIDataset(DATA_DIR, val_patients, transform=get_val_transform())
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    model = BrainClassifier(model_name=args.model, pretrained=True).to(DEVICE)
    
    # Class weights for BCE
    pos_weight = torch.tensor([2556.0 / 1373.0]).to(DEVICE) # Negative / Positive
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=MIXED_PRECISION)
    
    best_f1 = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': []}
    
    save_path = os.path.join(CHECKPOINT_DIR, "classification", f"{args.model}_best.pth")
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS} [{args.model}]")
        
        # Unfreeze backbone after 5 epochs
        if epoch == 1:
            print("Freezing backbone...")
            model.freeze_backbone()
        elif epoch == 6:
            print("Unfreezing backbone...")
            model.unfreeze_backbone()
            
        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
        val_loss, val_f1, val_acc, val_auc = evaluate(model, val_loader, criterion, DEVICE)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, best_f1, save_path)
            print("Saved best model!")
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print("Early stopping triggered!")
            break
            
    with open(os.path.join(CHECKPOINT_DIR, "training_logs", f"{args.model}_history.json"), "w") as f:
        json.dump(history, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=CLASSIFIERS)
    args = parser.parse_args()
    main(args)
