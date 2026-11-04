import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.utils import set_seed, get_patient_split, save_checkpoint, dice_coef_metric, iou_metric
from src.dataset import BrainMRIDataset, get_train_transform, get_val_transform
from src.models.segmentor import BrainSegmentor

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    # Only train on positive examples (masks with tumor)
    for images, masks, labels, _ in pbar:
        # Filter out negative examples
        pos_idx = (labels > 0).view(-1)
        if not pos_idx.any():
            continue
            
        images = images[pos_idx].to(device)
        masks = masks[pos_idx].to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type=device.type, enabled=MIXED_PRECISION):
            outputs = model(images)
            loss = criterion(outputs, masks)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        preds = torch.sigmoid(outputs).detach()
        running_dice += dice_coef_metric(preds, masks).item()
        running_iou += iou_metric(preds, masks).item()
        
        pbar.set_postfix({"loss": loss.item()})
        
    num_batches = len(dataloader) # approximate since we skip negatives
    epoch_loss = running_loss / max(1, num_batches)
    epoch_dice = running_dice / max(1, num_batches)
    epoch_iou = running_iou / max(1, num_batches)
    return epoch_loss, epoch_dice, epoch_iou

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    valid_batches = 0
    with torch.no_grad():
        for images, masks, labels, _ in tqdm(dataloader, desc="Evaluating"):
            pos_idx = (labels > 0).view(-1)
            if not pos_idx.any():
                continue
                
            valid_batches += 1
            images = images[pos_idx].to(device)
            masks = masks[pos_idx].to(device)
            
            with torch.amp.autocast(device_type=device.type, enabled=MIXED_PRECISION):
                outputs = model(images)
                loss = criterion(outputs, masks)
                
            running_loss += loss.item()
            
            preds = torch.sigmoid(outputs).detach()
            running_dice += dice_coef_metric(preds, masks).item()
            running_iou += iou_metric(preds, masks).item()
            
    if valid_batches == 0:
        return 0, 0, 0
        
    epoch_loss = running_loss / valid_batches
    epoch_dice = running_dice / valid_batches
    epoch_iou = running_iou / valid_batches
        
    return epoch_loss, epoch_dice, epoch_iou

def main(args):
    set_seed(SEED)
    
    train_patients, val_patients, _ = get_patient_split(DATA_DIR)
    
    train_dataset = BrainMRIDataset(DATA_DIR, train_patients, transform=get_train_transform())
    val_dataset = BrainMRIDataset(DATA_DIR, val_patients, transform=get_val_transform())
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    model = BrainSegmentor(architecture=args.arch, encoder_name=args.encoder).to(DEVICE)
    
    # Combined Loss: Dice + BCE
    # Using SMP losses
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce_loss = nn.BCEWithLogitsLoss()
    
    def combined_loss(y_pred, y_true):
        return dice_loss(y_pred, y_true) + bce_loss(y_pred, y_true)
        
    criterion = combined_loss
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=MIXED_PRECISION)
    
    best_dice = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_dice': [], 'train_iou': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}
    
    save_name = f"{args.arch}_{args.encoder}_best.pth"
    save_path = os.path.join(CHECKPOINT_DIR, "segmentation", save_name)
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS} [{args.arch} - {args.encoder}]")
        
        if epoch == 1:
            print("Freezing encoder...")
            model.freeze_encoder()
        elif epoch == 6:
            print("Unfreezing encoder...")
            model.unfreeze_encoder()
            
        train_loss, train_dice, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, DEVICE)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, best_dice, save_path)
            print("Saved best model!")
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print("Early stopping triggered!")
            break
            
    with open(os.path.join(CHECKPOINT_DIR, "training_logs", f"{args.arch}_{args.encoder}_history.json"), "w") as f:
        json.dump(history, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--encoder", type=str, required=True)
    args = parser.parse_args()
    main(args)
