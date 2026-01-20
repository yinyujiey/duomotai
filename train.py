import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from dataset import CLIPDataset
from models import GatedMultimodalClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8  # 开启增强和门控后，建议稍微减小 batch
LR = 1e-4      # 分类层的学习率可以设大一点
EPOCHS = 10
TRAIN_TXT = 'project5/train.txt'
DATA_DIR = 'project5/data/'

def train():
    # 为了图像增强，我们需要手动划分 indices
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    df = pd.read_csv(TRAIN_TXT)
    train_idx, val_idx = train_test_split(range(len(df)), test_size=0.2, random_state=42)
    
    # 训练集开启 is_train=True (增强)，验证集 is_train=False (不增强)
    full_dataset_train = CLIPDataset(TRAIN_TXT, DATA_DIR, is_train=True)
    full_dataset_val = CLIPDataset(TRAIN_TXT, DATA_DIR, is_train=False)
    
    train_ds = torch.utils.data.Subset(full_dataset_train, train_idx)
    val_ds = torch.utils.data.Subset(full_dataset_val, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = GatedMultimodalClassifier().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            ids, mask = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE)
            pixels, labels = batch['pixel_values'].to(DEVICE), batch['label'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(ids, mask, pixels, mode='both')
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model_gated.pth')

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            ids, mask = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE)
            pixels, labels = batch['pixel_values'].to(DEVICE), batch['label'].to(DEVICE)
            outputs = model(ids, mask, pixels, mode='both')
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return correct / total

if __name__ == "__main__":
    train()