# === IMPORTS ===
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np

# === DEVICE SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# === TRANSFORMS ===
train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === DATASET LOADING ===
data_dir = "/content/data/Dataset_BUSI_with_GT"
dataset = datasets.ImageFolder(root=data_dir, transform=train_tfms)

# Split train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_ds.dataset.transform = train_tfms
val_ds.dataset.transform = val_tfms

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

print(f"Train samples: {len(train_ds)}")
print(f"Val samples: {len(val_ds)}")

# === MODEL SETUP ===
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
for param in model.features.parameters():
    param.requires_grad = True  # full fine-tuning

# Replace classifier for 3 classes
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
model = model.to(device)

# === LOSS & OPTIMIZER ===
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


# === TRAINING LOOP ===
EPOCHS = 30

for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
    model.train()
    train_loss, train_preds, train_labels = 0, [], []

    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_preds += out.argmax(1).cpu().tolist()
        train_labels += y.cpu().tolist()

    train_acc = accuracy_score(train_labels, train_preds)
    train_f1 = f1_score(train_labels, train_preds, average='weighted')

    # Validation
    model.eval()
    val_loss, val_preds, val_labels = 0, [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item()
            val_preds += out.argmax(1).cpu().tolist()
            val_labels += y.cpu().tolist()

    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='weighted')

    try:
        val_auc = roc_auc_score(torch.nn.functional.one_hot(torch.tensor(val_labels)), torch.nn.functional.one_hot(torch.tensor(val_preds)))
    except:
        val_auc = float('nan')

    scheduler.step(val_loss)

    print(f"Train Loss: {train_loss/len(train_loader):.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
    print(f"Val Loss:   {val_loss/len(val_loader):.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

# === SAVE MODEL ===
torch.save(model.state_dict(), "breast_cancer_model_v2.pth")
print("\n✅ Training complete. Model saved as 'breast_cancer_model_v2.pth'")
