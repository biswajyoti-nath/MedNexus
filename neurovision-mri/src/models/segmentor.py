import segmentation_models_pytorch as smp
import torch.nn as nn

class BrainSegmentor(nn.Module):
    def __init__(self, architecture="Unet", encoder_name="resnet34", encoder_weights="imagenet"):
        super(BrainSegmentor, self).__init__()
        
        self.architecture = architecture
        self.encoder_name = encoder_name
        
        if architecture == "Unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=1,
            )
        elif architecture == "UnetPlusPlus":
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=1,
            )
        else:
            raise ValueError(f"Architecture {architecture} not supported")
            
    def forward(self, x):
        return self.model(x)

    def freeze_encoder(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = True
