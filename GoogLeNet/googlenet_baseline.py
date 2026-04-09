import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

# ==============================
# REPRODUCIBILITY
# ==============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==============================
# DATASET
# ==============================
class AlzheimerDataset(Dataset):
    def __init__(self, data_dir, transform=None, num_per_class=50):
        self.data_dir = data_dir
        self.transform = transform

        self.classes = [
            'NonDemented',
            'VeryMildDemented',
            'MildDemented',
            'ModerateDemented'
        ]

        self.image_paths = []
        self.labels = []

        for i, cls in enumerate(self.classes):
            cls_dir = os.path.join(data_dir, cls)

            all_files = os.listdir(cls_dir)
            selected_files = random.sample(
                all_files,
                min(num_per_class, len(all_files))
            )

            for f in selected_files:
                self.image_paths.append(
                    os.path.join(cls_dir, f)
                )
                self.labels.append(i)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# ==============================
# TRANSFORM
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "dataset", "Data")

# ==============================
# MODEL
# ==============================
class GoogLeNetModel(nn.Module):
    def __init__(self, num_classes=4):
        super(GoogLeNetModel, self).__init__()

        self.model = models.googlenet(weights="DEFAULT")
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)


# ==============================
# TRAIN
# ==============================
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs=10,
    device='cpu'
):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        val_loss, _, _, _, _ = evaluate_model(
            model,
            val_loader,
            criterion,
            device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    return train_losses, val_losses


# ==============================
# EVALUATION
# ==============================
def evaluate_model(model, loader, criterion, device='cpu'):
    model.eval()

    running_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            running_loss += loss.item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        running_loss / len(loader),
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        None
    )


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    full_dataset = AlzheimerDataset(
        DATA_DIR,
        transform=transform
    )

    labels = full_dataset.labels

    train_idx, temp_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=42,
        stratify=[labels[i] for i in temp_idx]
    )

    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=8,
        shuffle=True
    )

    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=8,
        shuffle=False
    )

    test_loader = DataLoader(
        Subset(full_dataset, test_idx),
        batch_size=8,
        shuffle=False
    )

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    model = GoogLeNetModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_losses, val_losses = train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs=10,
    device=device
)

    _, y_true, y_pred, y_probs, _ = evaluate_model( 
        model,
        test_loader,
        criterion,
        device
    )

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    auc = roc_auc_score(
        y_true,
        y_probs,
        multi_class='ovr'
    )

    cm = confusion_matrix(y_true, y_pred)

    specificity_scores = []

    for i in range(len(cm)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)

        specificity_i = tn / (tn + fp)
        specificity_scores.append(specificity_i)

    specificity = np.mean(specificity_scores)

    print("\nGOOGLENET FINAL RESULTS")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro F1    : {f1:.4f}")
    print(f"AUC-ROC     : {auc:.4f}")
    print(f"Sensitivity : {recall:.4f}")
    print(f"Specificity : {specificity:.4f}")

    # ==============================
    # TRAINING LOSS CURVE
    # ==============================
    plt.figure(figsize=(15, 5))

    # Loss Curve
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.title("GoogLeNet Loss Curve", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    # Confusion Matrix
    plt.subplot(1, 3, 2)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )
    plt.title("Confusion Matrix", fontsize=12, fontweight='bold')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Metrics Bar Graph
    metric_names = [
        'Accuracy',
        'Macro F1',
        'AUC-ROC',
        'Sensitivity',
        'Specificity'
    ]

    metric_values = [
        acc,
        f1,
        auc,
        recall,
        specificity
    ]

    plt.subplot(1, 3, 3)
    bars = plt.bar(metric_names, metric_values)

    plt.ylim(0, 1.1)
    plt.title("Performance Metrics", fontsize=12, fontweight='bold')
    plt.ylabel("Score")
    plt.xticks(rotation=20)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.3f}",
            ha='center',
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig("googlenet_results.png", dpi=300)
    plt.show()

    # ==============================
    # SEPARATE METRICS BAR GRAPH
    # ==============================
    metric_names = [
        'Accuracy',
        'Macro F1',
        'AUC-ROC',
        'Sensitivity',
        'Specificity'
    ]

    metric_values = [
        acc,
        f1,
        auc,
        recall,
        specificity
    ]

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        metric_names,
        metric_values,
        edgecolor='black',
        linewidth=1.2
    )

    plt.ylim(0, 1.1)
    plt.ylabel("Score", fontsize=12)
    plt.title(
        "GoogLeNet Test Set Performance Metrics",
        fontsize=14,
        fontweight='bold'
    )

    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.xticks(rotation=15)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.3f}",
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig("googlenet_metrics_bar_graph.png", dpi=300)
    plt.show()