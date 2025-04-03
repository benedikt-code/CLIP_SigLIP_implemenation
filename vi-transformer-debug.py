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
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score, roc_auc_score

# Vision Transformer components
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=3, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Linear projection: Weight matrix for projection
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x):
        print(f"PatchEmbedding input shape: {x.shape}")  # (B, C, H, W)

        # Step 1: Split image into patches
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            "Image size must be divisible by the patch size."
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        print(f"Shape after unfolding into patches: {patches.shape}")  # (B, C, n_patches_h, n_patches_w, patch_size, patch_size)

        # Step 2: Flatten patches
        patches = patches.contiguous().view(B, C, -1, self.patch_size * self.patch_size)
        print(f"Shape after flattening patches: {patches.shape}")  # (B, C, n_patches, patch_size * patch_size)

        # Step 3: Merge channels
        patches = patches.permute(0, 2, 1, 3).contiguous().view(B, -1, C * self.patch_size * self.patch_size)
        print(f"Shape after merging channels: {patches.shape}")  # (B, n_patches, patch_size * patch_size * C)

        # Step 4: Linear projection
        x = self.proj(patches)
        print(f"PatchEmbedding output shape after linear projection: {x.shape}")  # (B, n_patches, embed_dim)
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
    def __init__(self, img_size=28, patch_size=7, in_channels=3, embed_dim=128, depth=1, num_heads=8, mlp_dim=256):
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
        print(f"VisionTransformer input shape: {x.shape}")
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        print(f"CLS token shape: {cls_tokens.shape}")
        x = torch.cat((cls_tokens, x), dim=1)
        print(f"Shape after concatenating CLS token: {x.shape}")
        x = x + self.pos_embed[:, :x.size(1), :]
        print(f"Shape after adding positional embeddings: {x.shape}")
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        print(f"VisionTransformer output shape: {x.shape}")
        return x[:, 0]  # return CLS token only

class BloodMNISTImagePairDataset(Dataset):
    def __init__(self, npz_path, split='train', transform=None):
        data = np.load(npz_path)
        self.images = data[f"{split}_images"]  # Load only 3 images
        self.labels = data[f"{split}_labels"].squeeze()  # Load only 3 labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_array = (self.images[idx] * 255).astype(np.uint8)
        img = Image.fromarray(img_array).convert("RGB")
        augmented = self.transform(img) if self.transform else img

        img1 = transforms.ToTensor()(img)
        img2 = transforms.ToTensor()(augmented)
        print(f"Dataset sample {idx}: Original shape {img1.shape}, Augmented shape {img2.shape}")
        return img1, img2, self.labels[idx]

def train_vit_debug():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ Device:", device)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
    ])

    dataset = BloodMNISTImagePairDataset("./data/bloodmnist.npz", split="train", transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    model = VisionTransformer()
    model = model.to(device)

    for img1, img2, label in dataloader:
        img1, img2 = img1.to(device), img2.to(device)
        print(f"Input image 1 shape: {img1.shape}")
        print(f"Input image 2 shape: {img2.shape}")

        with autocast():
            z1 = F.normalize(model(img1), dim=-1)
            z2 = F.normalize(model(img2), dim=-1)
            print(f"Normalized output z1 shape: {z1.shape}")
            print(f"Normalized output z2 shape: {z2.shape}")

            logits = z1 @ z2.T * 100.0
            print(f"Logits shape: {logits.shape}")
            print(f"Logits: {logits}")

if __name__ == "__main__":
    train_vit_debug()