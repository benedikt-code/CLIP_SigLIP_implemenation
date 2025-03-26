import medmnist
from medmnist import INFO, BloodMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

# Lade Dataset-Infos
data_flag = 'bloodmnist'
download = True

# Optional: Infos anzeigen
print(INFO[data_flag])

# Transformation definieren (z. B. Tensor-Konvertierung)
data_transforms = transforms.Compose([
    transforms.ToTensor()
])

# Training & Testdaten laden
train_dataset = BloodMNIST(split='train', transform=data_transforms, download=download)
test_dataset = BloodMNIST(split='test', transform=data_transforms, download=download)

# DataLoader zum Durchlaufen
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
