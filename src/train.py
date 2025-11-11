import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils import compute_metrics
from dataset import get_dataloaders
from model import build_model

def train_model(data_dir, epochs=30, batch_size=16, lr=1e-4, weight_decay=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load data
    train_loader, val_loader = get_dataloaders(data_dir, batch_size)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Model, loss, optimizer
    model = build_model(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
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

        train_acc, train_f1, _ = compute_metrics(train_labels, train_preds)

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

        val_acc, val_f1, val_auc = compute_metrics(val_labels, val_preds)
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val Loss:   {val_loss/len(val_loader):.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

    torch.save(model.state_dict(), "breast_cancer_model_v2.pth")
    print("\n✅ Training complete. Model saved as 'breast_cancer_model_v2.pth'")
