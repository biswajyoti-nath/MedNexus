import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

def generate_agreement_map(masks_list):
    """
    masks_list: list of binary numpy masks (H, W) from different segmentors.
    Returns an agreement map: (H, W) where values range from 0 to len(masks_list).
    """
    agreement = np.zeros_like(masks_list[0], dtype=np.float32)
    for mask in masks_list:
        agreement += mask
        
    return agreement

def visualize_agreement_map(img_np, agreement_map, num_models, save_path=None):
    """
    Visualizes agreement map with distinct colors.
    Green = high agreement (all models predicted tumor)
    Red/Yellow = low agreement (only 1 or some models predicted tumor)
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(img_np)
    
    # Create an RGBA overlay
    overlay = np.zeros((*agreement_map.shape, 4))
    
    # Define colors based on agreement level (assuming 3 models)
    if num_models == 3:
        overlay[agreement_map == 1] = [1, 0, 0, 0.5]     # Red (1 model)
        overlay[agreement_map == 2] = [1, 1, 0, 0.5]     # Yellow (2 models)
        overlay[agreement_map == 3] = [0, 1, 0, 0.5]     # Green (3 models)
    else:
        # Generic colormap
        norm_agreement = agreement_map / num_models
        overlay[agreement_map > 0] = plt.cm.jet(norm_agreement[agreement_map > 0])
        overlay[..., 3] = 0.5 * (agreement_map > 0)
        
    plt.imshow(overlay)
    plt.title("Model Agreement Map\nRed: Low Agreement, Green: High Agreement")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def mc_dropout_uncertainty(model, img_tensor, num_samples=10):
    """
    Enable dropout at inference time to compute uncertainty (variance).
    Returns mean prediction and variance map.
    """
    model.train() # Enable dropout
    preds = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            out = model(img_tensor)
            preds.append(torch.sigmoid(out).cpu().numpy())
            
    preds = np.array(preds)
    mean_pred = np.mean(preds, axis=0).squeeze()
    variance_map = np.var(preds, axis=0).squeeze()
    
    model.eval() # Restore eval mode
    return mean_pred, variance_map

def visualize_uncertainty(img_np, variance_map, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title("Original MRI")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(variance_map, cmap='hot')
    plt.title("Uncertainty (MC Dropout Variance)")
    plt.colorbar()
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
