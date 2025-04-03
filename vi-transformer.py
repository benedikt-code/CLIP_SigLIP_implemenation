# Vision Transformer (ViT) with contrastive learning on BloodMNIST + Evaluation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import math
import os
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# Vision Transformer components
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=3, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = attn_scores.softmax(dim=-1)
        attn_output = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(attn_output)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=3, embed_dim=128, depth=6, num_heads=8, mlp_dim=256):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.dropout = nn.Dropout(0.1)

        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]  # return CLS token only

class BloodMNISTImagePairDataset(Dataset):
    def __init__(self, npz_path, split='train', transform=None):
        data = np.load(npz_path)
        self.images = data[f"{split}_images"]
        self.labels = data[f"{split}_labels"].squeeze()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_array = (self.images[idx] * 255).astype(np.uint8)
        img = Image.fromarray(img_array).convert("RGB")
        augmented = self.transform(img) if self.transform else img

        img1 = transforms.ToTensor()(img)
        img2 = transforms.ToTensor()(augmented)
        return img1, img2, self.labels[idx]

def info_nce_loss(logits):
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size, device=logits.device)
    return F.cross_entropy(logits, labels)

def evaluate(model, npz_path, split, device):
    data = np.load(npz_path)
    images = data[f"{split}_images"]
    labels = data[f"{split}_labels"].squeeze()

    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(images), 64):
            batch_imgs = []
            batch_labels = []
            for j in range(i, min(i + 64, len(images))):
                img = Image.fromarray((images[j] * 255).astype(np.uint8)).convert("RGB")
                tensor = transforms.ToTensor()(img)
                batch_imgs.append(tensor)
                batch_labels.append(labels[j])

            batch_tensor = torch.stack(batch_imgs).to(device)
            batch_embeddings = model(batch_tensor)
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

def train_vit_contrastive():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ Device:", device)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
    ])

    dataset = BloodMNISTImagePairDataset("./data/bloodmnist.npz", split="train", transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)

    model_1 = VisionTransformer()
    model_2 = VisionTransformer()

    if torch.cuda.device_count() > 1:
        model_1 = nn.DataParallel(model_1)
        model_2 = nn.DataParallel(model_2)

    model_1 = model_1.to(device)
    model_2 = model_2.to(device)

    optimizer = optim.Adam(list(model_1.parameters()) + list(model_2.parameters()), lr=1e-4)
    scaler = torch.amp.GradScaler()

    for epoch in range(1, 41):
        model_1.train()
        model_2.train()
        total_loss = 0

        for img1, img2, _ in tqdm(dataloader, desc=f"Epoch {epoch}"):
            img1, img2 = img1.to(device), img2.to(device)

            optimizer.zero_grad()
            with autocast():
                z1 = F.normalize(model_1(img1), dim=-1)
                z2 = F.normalize(model_2(img2), dim=-1)

                logits = z1 @ z2.T * 100.0
                loss = info_nce_loss(logits)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"✅ Epoch {epoch} Loss: {total_loss / len(dataloader):.4f}")

        # Evaluation
        eval_model = model_1.module if isinstance(model_1, nn.DataParallel) else model_2
        acc_train, auc_train = evaluate(eval_model, "./data/bloodmnist.npz", split="train", device=device)
        acc_val, auc_val = evaluate(eval_model, "./data/bloodmnist.npz", split="val", device=device)
        acc_test, auc_test = evaluate(eval_model, "./data/bloodmnist.npz", split="test", device=device)

        print(f"\n📊 Results after Epoch {epoch}:")
        print(f"Train  Accuracy: {acc_train:.4f} | AUC: {auc_train:.4f}")
        print(f"Val    Accuracy: {acc_val:.4f} | AUC: {auc_val:.4f}")
        print(f"Test   Accuracy: {acc_test:.4f} | AUC: {auc_test:.4f}\n")

if __name__ == "__main__":
    train_vit_contrastive()
