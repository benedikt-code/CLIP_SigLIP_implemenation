# Vision Transformer (ViT) with contrastive learning on BloodMNIST + Evaluation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import math
import os
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# Vision Transformer components
# Does all the patching - splitting the image into patches and projecting them
# --> using a linear layer (Conv2d would be an option too)
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=3, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Linear projection: Weight matrix for projection
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x):
        #print(f"PatchEmbedding input shape: {x.shape}")  # (B, C, H, W)

        # Step 1: Split image into patches
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            "Image size must be divisible by the patch size."
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        #print(f"Shape after unfolding into patches: {patches.shape}")  # (B, C, n_patches_h, n_patches_w, patch_size, patch_size)

        # Step 2: Flatten patches
        patches = patches.contiguous().view(B, C, -1, self.patch_size * self.patch_size)
        #print(f"Shape after flattening patches: {patches.shape}")  # (B, C, n_patches, patch_size * patch_size)

        # Step 3: Merge channels
        patches = patches.permute(0, 2, 1, 3).contiguous().view(B, -1, C * self.patch_size * self.patch_size)
        #print(f"Shape after merging channels: {patches.shape}")  # (B, n_patches, patch_size * patch_size * C)

        # Step 4: Linear projection
        x = self.proj(patches)
        #print(f"PatchEmbedding output shape after linear projection: {x.shape}")  # (B, n_patches, embed_dim)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0

        #qkv values
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        #Variables B = batch size, N = number of patches, C = embedding dimension
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        #qkv is split
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        #attention scores are calculated by multiplying q and k, then softmax is applied to get attention probabilities
        #and finally the attention probabilities are multiplied with v to get the final output
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

        #Transformer blocks, each block consists of a multi-head self-attention layer and a feedforward neural network
        # --> each block is followed by a layer normalization
        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    #Does the forward pass of the model
    # --> input is an image, output is the embedding of the CLS token    
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

#Loads Bloodmnist dataset from npz file and applies transformations
# --> returns two augmented images and the label
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
    """
    if torch.cuda.device_count() > 1:
        model_1 = nn.DataParallel(model_1)
        model_2 = nn.DataParallel(model_2)
    """
    model_1 = model_1.to(device)
    model_2 = model_2.to(device)

    optimizer = optim.Adam(list(model_1.parameters()) + list(model_2.parameters()), lr=1e-4)
    scaler = GradScaler()

    # Lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    train_aucs = []
    val_accuracies = []
    val_aucs = []
    test_accuracies = []
    test_aucs = []

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
                #Temp value = 100
                logits = z1 @ z2.T * 100.0
                loss = info_nce_loss(logits)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"✅ Epoch {epoch} Loss: {avg_loss:.4f}")

        # Evaluation
        eval_model = model_1.module if isinstance(model_1, nn.DataParallel) else model_1
        acc_train, auc_train = evaluate(eval_model, "./data/bloodmnist.npz", split="train", device=device)
        acc_val, auc_val = evaluate(eval_model, "./data/bloodmnist.npz", split="val", device=device)
        acc_test, auc_test = evaluate(eval_model, "./data/bloodmnist.npz", split="test", device=device)

        train_accuracies.append(acc_train)
        train_aucs.append(auc_train)
        val_accuracies.append(acc_val)
        val_aucs.append(auc_val)
        test_accuracies.append(acc_test)
        test_aucs.append(auc_test)

        print(f"\n📊 Results after Epoch {epoch}:")
        print(f"Train  Accuracy: {acc_train:.4f} | AUC: {auc_train:.4f}")
        print(f"Val    Accuracy: {acc_val:.4f} | AUC: {auc_val:.4f}")
        print(f"Test   Accuracy: {acc_test:.4f} | AUC: {auc_test:.4f}\n")

    # Save the model parameters
    torch.save(model_1.state_dict(), "vit_model_1.pth")
    torch.save(model_2.state_dict(), "vit_model_2.pth")
    print("💾 Model parameters saved as 'vit_model_1.pth' and 'vit_model_2.pth'")

    # Plot and save metrics
    plt.figure(figsize=(12, 8))

    # Plot Loss
    plt.subplot(3, 1, 1)
    plt.plot(range(1, 41), train_losses, label="Train Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(3, 1, 2)
    plt.plot(range(1, 41), train_accuracies, label="Train Accuracy", color="green")
    plt.plot(range(1, 41), val_accuracies, label="Validation Accuracy", color="orange")
    plt.plot(range(1, 41), test_accuracies, label="Test Accuracy", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    # Plot AUC
    plt.subplot(3, 1, 3)
    plt.plot(range(1, 41), train_aucs, label="Train AUC", color="green")
    plt.plot(range(1, 41), val_aucs, label="Validation AUC", color="orange")
    plt.plot(range(1, 41), test_aucs, label="Test AUC", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("AUC")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    print("📊 Training metrics plot saved as 'training_metrics.png'")

if __name__ == "__main__":
    train_vit_contrastive()
