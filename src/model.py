import torch
import torch.nn as nn
import torchvision.models as models

class SARWindResNet(nn.Module):
    def __init__(self):
        super(SARWindResNet, self).__init__()
        
        # Load a pre-built ResNet18 model
        self.resnet = models.resnet18(pretrained=False)
        
        # Modify the first layer: SAR images are 1-channel (Grayscale), not 3-channel (RGB)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the final layer: We want 2 continuous outputs (U and V vectors), not 1000 classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 2) # Output: [dx, dy]
        )

    def forward(self, x):
        return self.resnet(x)

if __name__ == "__main__":
    # Test the model with a fake 256x256 SAR image patch
    model = SARWindResNet()
    dummy_sar_image = torch.randn(1, 1, 256, 256)
    predicted_wind_vectors = model(dummy_sar_image)
    print(f"Model successfully output vector [U, V]: {predicted_wind_vectors.detach().numpy()}")