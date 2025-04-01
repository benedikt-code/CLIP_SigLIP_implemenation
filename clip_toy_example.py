import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_paths = ["./data/pics/IMG_1.jpg", "./data/pics/IMG_2.jpg", "./data/pics/IMG_3.jpg"]
images = [preprocess(Image.open(img)).unsqueeze(0).to(device) for img in image_paths]
images = torch.cat(images)  # Stack images into a batch

labels = ["a confused beautiful man", "club mate in front of sunlit buildings", "two headphones lying down"] 
text = clip.tokenize(labels).to(device)

with torch.no_grad():
    image_features = model.encode_image(images)
    text_features = model.encode_text(text)

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

similarity = (image_features @ text_features.T).softmax(dim=-1)*100.0

for i, img_path in enumerate(image_paths):
    print(f"Image: {img_path}")
    for j, label in enumerate(labels):
        print(f"  {label}: {similarity[i, j].item():.4f}")

