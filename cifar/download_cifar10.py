import torchvision
import warnings
import os

# Ignore warning messages
warnings.filterwarnings("ignore")

# Create data directory if it doesn't exist
os.makedirs('./data', exist_ok=True)

# Download the training dataset
print("Downloading CIFAR10 training dataset...")
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=None
)

# Download the test dataset
print("Downloading CIFAR10 test dataset...")
test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=None
)

print(f"CIFAR10 dataset downloaded successfully!")
print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")

import torch
from torchvision.models import resnet50, ResNet50_Weights
import os

# Create directory to store weights
os.makedirs('./model-weights', exist_ok=True)

# Download pretrained ResNet50 weights
print("Downloading pretrained ResNet50 weights...")
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# Save weights locally
weights_path = './model-weights/resnet50_weights.pth'
torch.save(model.state_dict(), weights_path)

print(f"ResNet50 pretrained weights saved successfully at {weights_path}")