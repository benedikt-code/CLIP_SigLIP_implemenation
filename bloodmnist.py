from medmnist import BloodMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

# Custom root directory for the dataset
data_root = './data'

# Transformation (convert images to tensor)
data_transforms = transforms.Compose([
    transforms.ToTensor()
])

# Load train & test splits from local file
train_dataset = BloodMNIST(split='train', transform=data_transforms, download=False, root=data_root)
test_dataset = BloodMNIST(split='test', transform=data_transforms, download=False, root=data_root)

# Optional: Wrap in DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
