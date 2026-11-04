import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "kaggle_3m")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure directories exist
os.makedirs(os.path.join(CHECKPOINT_DIR, "classification"), exist_ok=True)
os.makedirs(os.path.join(CHECKPOINT_DIR, "segmentation"), exist_ok=True)
os.makedirs(os.path.join(CHECKPOINT_DIR, "training_logs"), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hardware setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIXED_PRECISION = True

# Hyperparameters
IMG_SIZE = 256
BATCH_SIZE = 16
NUM_WORKERS = 4
LR = 1e-4
EPOCHS = 50
PATIENCE = 10
SEED = 42

# Models to train
CLASSIFIERS = ["resnet34", "efficientnet_b0", "densenet121"]
SEGMENTORS = [
    {"arch": "Unet", "encoder": "resnet34"},
    {"arch": "Unet", "encoder": "efficientnet-b0"},
    {"arch": "UnetPlusPlus", "encoder": "densenet121"}
]
