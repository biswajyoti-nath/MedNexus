import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook the target layer
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x):
        self.model.eval()
        output = self.model(x)
        
        # We want to explain the probability of tumor (class 1)
        self.model.zero_grad()
        output.backward(retain_graph=True)
        
        # Global average pooling on gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # ReLU to keep only positive influence
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().detach().numpy()
        
def show_cam_on_image(img_np, cam, save_path=None):
    """
    img_np: numpy array (H, W, 3), float [0, 1]
    cam: numpy array (H, W), float [0, 1]
    """
    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    cam_result = heatmap + np.float32(img_np)
    cam_result = cam_result / np.max(cam_result)
    
    plt.imshow(cam_result)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
