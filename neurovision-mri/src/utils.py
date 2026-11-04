import os
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from config import SEED

def set_seed(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_patient_split(data_dir, val_size=0.1, test_size=0.1, seed=SEED):
    """
    Splits data at the patient level to prevent data leakage across slices.
    """
    patients = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    
    # Train / (Val + Test)
    train_patients, temp_patients = train_test_split(patients, test_size=(val_size + test_size), random_state=seed)
    
    # Val / Test
    val_p = val_size / (val_size + test_size)
    val_patients, test_patients = train_test_split(temp_patients, test_size=(1.0 - val_p), random_state=seed)
    
    return train_patients, val_patients, test_patients

def dice_coef_metric(y_pred, y_true, smooth=1e-5):
    y_pred = (y_pred > 0.5).float()
    intersection = (y_pred * y_true).sum()
    return (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

def iou_metric(y_pred, y_true, smooth=1e-5):
    y_pred = (y_pred > 0.5).float()
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def save_checkpoint(model, optimizer, epoch, best_metric, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
    }, path)

def load_checkpoint(model, path, optimizer=None):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['best_metric']
