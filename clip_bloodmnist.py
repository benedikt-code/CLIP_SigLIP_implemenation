from torchvision import transforms
from PIL import Image
import os
import random
import clip
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Transformationen definieren
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Spiegeln
    transforms.RandomRotation(30),      # Drehung
    transforms.ToTensor()
])

# Beispiel für das Laden und Transformieren eines Bildes # benutze ich nicht
# def load_and_transform_image(image_path):
#     img = Image.open(image_path).convert('RGB')
#    transformed_img = transform(img)
#    return transformed_img

# Erstelle ein Paar aus Original und transformiertem Bild
def create_image_pair(image_path):
    original_image = Image.open(image_path).convert('RGB')
    transformed_image = transform(original_image)
    return original_image, transformed_image

# Load the dataset
data = np.load("./data/bloodmnist.npz")
images = data["images"]

# Convert a single image to PIL and apply transformation
def process_image(image_array):
    img = Image.fromarray((image_array * 255).astype(np.uint8))  # Convert to 8-bit image
    transformed_img = transform(img)
    return transformed_img

# Beispiel, um Bildpaare zu erstellen
image_dir = './data'  # Pfad zu deinem BloodMNIST Ordner
image_files = os.listdir(image_dir)

image_pairs = []
for img_file in image_files:
    img_path = os.path.join(image_dir, img_file)
    original, transformed = create_image_pair(img_path)
    image_pairs.append((original, transformed))

class BloodMNISTDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        original = Image.fromarray((self.images[idx] * 255).astype(np.uint8))  # Convert to PIL
        transformed = self.transform(original) if self.transform else original
        original = transforms.ToTensor()(original)
        return original, transformed

# Create dataset and dataloader
dataset = BloodMNISTDataset(images, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Lade das CLIP-Modell
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# Erstelle einen benutzerdefinierten Dataset-Klasse
class ImagePairDataset(Dataset):
    def __init__(self, image_pairs, transform=None):
        self.image_pairs = image_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        original, transformed = self.image_pairs[idx]
        original_tensor = preprocess(original)
        transformed_tensor = preprocess(transformed)
        return original_tensor, transformed_tensor

# Erstelle den Dataset und DataLoader
dataset = ImagePairDataset(image_pairs)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Definiere den Optimierer
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Training
for epoch in range(10):  # Anzahl der Epochen
    model.train()
    for i, (images1, images2) in enumerate(dataloader):
        images1 = images1.to(device)
        images2 = images2.to(device)

        # Berechne die Bildmerkmale
        image_features1 = model.encode_image(images1)
        image_features2 = model.encode_image(images2)

        # Berechne den Verlust (Contrastive Loss)
        similarity = torch.cosine_similarity(image_features1, image_features2)
        loss = -similarity.mean()  # Minimieren der negativen Ähnlichkeit

        # Backpropagation und Optimierung
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')