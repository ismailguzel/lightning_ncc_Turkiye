import torchvision
import warnings
import os

# Ignore warning messages
warnings.filterwarnings("ignore")

# Create data directory if it doesn't exist
os.makedirs('./data', exist_ok=True)

# Download the training dataset
print("Downloading MNIST training dataset...")
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=None
)

# Download the test dataset
print("Downloading MNIST test dataset...")
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=None
)

print(f"MNIST dataset downloaded successfully!")
print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")
