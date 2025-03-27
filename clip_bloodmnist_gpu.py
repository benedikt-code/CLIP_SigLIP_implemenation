from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import clip
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, roc_auc_score

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
        return img1, img2, self.labels[idx]

def info_nce_loss(logits):
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size, device=logits.device)
    return F.cross_entropy(logits, labels)

def evaluate(model, npz_path, split, clip_preprocess, device):
    data = np.load(npz_path)
    images = data[f"{split}_images"]
    labels = data[f"{split}_labels"].squeeze()

    model.eval()
    encode_fn = model

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

            batch_tensor = torch.stack(batch_imgs).to(device)
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

def train_clip_dual_encoder():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device} | GPU count: {torch.cuda.device_count()}")

    base_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False, download_root="./clip_model_cache")
    encoder_1 = base_model.visual
    encoder_2 = clip.load("ViT-B/32", device=device, jit=False, download_root="./clip_model_cache")[0].visual

    encoder_1 = encoder_1.float()
    encoder_2 = encoder_2.float()

    if torch.cuda.device_count() > 1:
        encoder_1 = torch.nn.DataParallel(encoder_1)
        encoder_2 = torch.nn.DataParallel(encoder_2)

    encoder_1 = encoder_1.to(device)
    encoder_2 = encoder_2.to(device)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
    ])

    dataset = BloodMNISTImagePairDataset(
        npz_path="./data/bloodmnist.npz",
        split="train",
        transform=transform,
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

    optimizer = optim.Adam(list(encoder_1.parameters()) + list(encoder_2.parameters()), lr=1e-5)
    scaler = GradScaler()

    for epoch in range(1, 11):
        encoder_1.train()
        encoder_2.train()
        total_loss = 0

        for img1, img2, _ in tqdm(dataloader, desc=f"Epoch {epoch}"):
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                feat1 = encoder_1(img1)
                feat2 = encoder_2(img2)

                feat1 = F.normalize(feat1, dim=-1)
                feat2 = F.normalize(feat2, dim=-1)

                logits = feat1 @ feat2.T * 100.0
                loss = info_nce_loss(logits)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")

        # Evaluation after epoch
        eval_model = encoder_1.module if isinstance(encoder_1, torch.nn.DataParallel) else encoder_1
        acc_train, auc_train = evaluate(eval_model, "./data/bloodmnist.npz", split="train", clip_preprocess=clip_preprocess, device=device)
        acc_val, auc_val = evaluate(eval_model, "./data/bloodmnist.npz", split="val", clip_preprocess=clip_preprocess, device=device)
        acc_test, auc_test = evaluate(eval_model, "./data/bloodmnist.npz", split="test", clip_preprocess=clip_preprocess, device=device)

        print(f"\n📊 Results after Epoch {epoch}:")
        print(f"Train Accuracy: {acc_train:.4f} | AUC: {auc_train:.4f}")
        print(f"Val   Accuracy: {acc_val:.4f} | AUC: {auc_val:.4f}")
        print(f"Test  Accuracy: {acc_test:.4f} | AUC: {auc_test:.4f}\n")

        torch.save({
            "encoder_1": encoder_1.module.state_dict() if isinstance(encoder_1, torch.nn.DataParallel) else encoder_1.state_dict(),
            "encoder_2": encoder_2.module.state_dict() if isinstance(encoder_2, torch.nn.DataParallel) else encoder_2.state_dict()
        }, f"clip_dual_encoder_epoch{epoch}.pth")
        print("💾 Saved model checkpoint\n")

if __name__ == "__main__":
    train_clip_dual_encoder()
