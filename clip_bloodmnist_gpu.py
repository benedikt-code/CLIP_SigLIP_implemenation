from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import clip
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
import os

# Dataset class for original + augmented image pairs
class BloodMNISTImagePairDataset(Dataset):
    def __init__(self, npz_path, split='train', transform=None, clip_preprocess=None):
        data = np.load(npz_path)
        self.images = data[f"{split}_images"]
        self.labels = data[f"{split}_labels"].squeeze()
        self.transform = transform
        self.clip_preprocess = clip_preprocess

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_array = (self.images[idx] * 255).astype(np.uint8)
        img = Image.fromarray(img_array).convert("RGB")

        augmented = self.transform(img) if self.transform else img

        img1 = self.clip_preprocess(img)
        img2 = self.clip_preprocess(augmented)

        return img1, img2

# Evaluation function using class means + cosine similarity
def evaluate(model, clip_preprocess, npz_path, split, device):
    data = np.load(npz_path)
    images = data[f"{split}_images"]
    labels = data[f"{split}_labels"].squeeze()

    model.eval()
    encode_fn = model.module.encode_image if isinstance(model, torch.nn.DataParallel) else model.encode_image

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(images), 64):
            batch_imgs = []
            batch_labels = []

            for j in range(i, min(i + 64, len(images))):
                img = Image.fromarray((images[j] * 255).astype(np.uint8)).convert("RGB")
                tensor = clip_preprocess(img)
                batch_imgs.append(tensor)
                batch_labels.append(labels[j])

            batch_tensor = torch.stack(batch_imgs).to(device, non_blocking=True)
            batch_embeddings = encode_fn(batch_tensor)
            all_embeddings.append(batch_embeddings)
            all_labels.extend(batch_labels)

    all_embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()
    all_labels = np.array(all_labels)

    class_means = []
    for class_idx in np.unique(all_labels):
        class_mean = all_embeddings[all_labels == class_idx].mean(axis=0)
        class_means.append(class_mean)
    class_means = np.stack(class_means)

    logits = all_embeddings @ class_means.T
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(all_labels, preds)

    y_true = np.eye(len(class_means))[all_labels]
    y_score = logits
    try:
        auc = roc_auc_score(y_true, y_score, multi_class="ovr")
    except ValueError:
        auc = float('nan')

    return acc, auc

# Main training function
def train_clip_encoder_on_pairs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device} | GPU count: {torch.cuda.device_count()}")

    model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False, download_root="./clip_model_cache")

    model = model.float()

    if torch.cuda.device_count() > 1:
        print("🚀 Using DataParallel for multi-GPU training")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    custom_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
    ])

    dataset = BloodMNISTImagePairDataset(
        npz_path="./data/bloodmnist.npz",
        split="train",
        transform=custom_transform,
        clip_preprocess=clip_preprocess
    )

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=8,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    model.train()

    best_val_auc = 0.0

    for epoch in range(1, 11):
        total_loss = 0
        for img1, img2 in tqdm(dataloader, desc=f"Epoch {epoch}"):
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)

            f1 = model.module.encode_image(img1) if isinstance(model, torch.nn.DataParallel) else model.encode_image(img1)
            f2 = model.module.encode_image(img2) if isinstance(model, torch.nn.DataParallel) else model.encode_image(img2)

            similarity = F.cosine_similarity(f1, f2, dim=-1)
            loss = -similarity.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")

        with torch.no_grad():
            train_acc, train_auc = evaluate(model, clip_preprocess, "./data/bloodmnist.npz", split="train", device=device)
            val_acc, val_auc = evaluate(model, clip_preprocess, "./data/bloodmnist.npz", split="val", device=device)
            test_acc, test_auc = evaluate(model, clip_preprocess, "./data/bloodmnist.npz", split="test", device=device)

        print(f"\n📊 Results after Epoch {epoch}:")
        print(f"Train  Accuracy: {train_acc:.4f} | AUC: {train_auc:.4f}")
        print(f"Val    Accuracy: {val_acc:.4f} | AUC: {val_auc:.4f}")
        print(f"Test   Accuracy: {test_acc:.4f} | AUC: {test_auc:.4f}\n")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), f"clip_model_best.pth")
            print("💾 Saved new best model!")

        model.train()

if __name__ == "__main__":
    train_clip_encoder_on_pairs()
