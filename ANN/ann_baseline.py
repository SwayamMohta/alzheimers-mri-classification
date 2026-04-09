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
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)
import pandas as pd

# =============================================================================
# 1. REPRODUCIBILITY (FIXING SEEDS)
# =============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# =============================================================================
# 2. DATASET DEFINITION & PREPROCESSING
# =============================================================================
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

        # Load exactly 50 images per class using RANDOM SAMPLING
        for i, cls in enumerate(self.classes):
            cls_dir = os.path.join(data_dir, cls)

            if not os.path.exists(cls_dir):
                print(f"Warning: Directory {cls_dir} not found. Skipping.")
                continue

            all_files = sorted(os.listdir(cls_dir))
            selected_files = random.sample(
                all_files,
                min(num_per_class, len(all_files))
            )

            for f in selected_files:
                self.image_paths.append(os.path.join(cls_dir, f))
                self.labels.append(i)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # Flatten image to 1D vector
        image = image.view(-1)

        return image, label

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "dataset", "Data")

# =============================================================================
# 3. EARLY STOPPING
# =============================================================================
class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1

            if self.verbose:
                print(
                    f"EarlyStopping counter: "
                    f"{self.counter} out of {self.patience}"
                )

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f"Validation loss decreased "
                f"({self.val_loss_min:.6f} --> {val_loss:.6f}). "
                f"Saving model..."
            )

        torch.save(model.state_dict(), 'best_ann_model.pth')
        self.val_loss_min = val_loss

# =============================================================================
# 4. MODEL ARCHITECTURE
# =============================================================================
class ANNModel(nn.Module):
    def __init__(
        self,
        input_dim=16384,
        hidden1=512,
        hidden2=128,
        output_dim=4
    ):
        super(ANNModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# =============================================================================
# 5. TRAINING
# =============================================================================
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs=20,
    device='cpu'
):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    early_stopping = EarlyStopping(patience=3, verbose=True)

    print("-" * 30)
    print("STARTING TRAINING LOOP")
    print("-" * 30)

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = (correct / total) * 100

        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        val_loss, val_acc, _, _, _ = evaluate_model(
            model,
            val_loader,
            criterion,
            device
        )

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch [{epoch+1:2d}/{epochs}] "
            f"Loss: {epoch_loss:.4f} | "
            f"Acc: {epoch_acc:5.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:5.2f}%"
        )

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    return train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, loader, criterion, device='cpu'):
    model.eval()

    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        running_loss / len(loader.dataset),
        (correct / total) * 100,
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )

# =============================================================================
# 6. MAIN FLOW
# =============================================================================
if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} directory not found.")

    else:
        full_dataset = AlzheimerDataset(DATA_DIR, transform=transform)
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
            batch_size=16,
            shuffle=True
        )

        val_loader = DataLoader(
            Subset(full_dataset, val_idx),
            batch_size=16,
            shuffle=False
        )

        test_loader = DataLoader(
            Subset(full_dataset, test_idx),
            batch_size=16,
            shuffle=False
        )

        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        model = ANNModel().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses, val_losses, train_accs, val_accs = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            epochs=20,
            device=device
        )

        model.load_state_dict(torch.load('best_ann_model.pth'))

        _, _, y_true, y_pred, y_probs = evaluate_model(
            model,
            test_loader,
            criterion,
            device
        )

        # ==============================
        # FINAL METRICS CALCULATION
        # ==============================
        acc = accuracy_score(y_true, y_pred)

        prec = precision_score(
            y_true,
            y_pred,
            average='macro',
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            average='macro',
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average='macro'
        )

        try:
            auc = roc_auc_score(
                y_true,
                y_probs,
                multi_class='ovr'
            )
        except ValueError:
            auc = float('nan')

        cm = confusion_matrix(y_true, y_pred)

        # ==============================
        # ADDITIONAL METRICS
        # ==============================
        # Sensitivity = Recall (macro average)
        sensitivity = recall

        # Specificity calculation for multi-class classification
        specificity_scores = []

        for i in range(len(cm)):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - (tp + fn + fp)

            specificity_i = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_scores.append(specificity_i)

        specificity = np.mean(specificity_scores)

        # ==============================
        # CLEAN FINAL OUTPUT
        # ==============================
        print("\n" + "=" * 40)
        print("ANN FINAL PERFORMANCE METRICS")
        print("=" * 40)

        print(f"Accuracy    : {acc:.4f}")
        print(f"Precision   : {prec:.4f}")
        print(f"Recall      : {recall:.4f}")
        print(f"F1 Score    : {f1:.4f}")
        print(f"AUC-ROC     : {auc:.4f}")
        print(f"Sensitivity : {sensitivity:.4f}")
        print(f"Specificity : {specificity:.4f}")

        # ==============================
        # TRAINING CURVES + CONFUSION MATRIX
        # ==============================
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Val')
        plt.title('Loss Curve')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(train_accs, label='Train')
        plt.plot(val_accs, label='Val')
        plt.title('Accuracy Curve')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()

        plt.subplot(1, 3, 3)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues'
        )
        plt.title('Confusion Matrix')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.tight_layout()
        plt.savefig('ann_results.png')
        plt.show()

        # ==============================
        # FINAL METRICS GRAPH (IMPROVED)
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
            sensitivity,
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
        plt.title("ANN Test Set Performance Metrics", fontsize=14, fontweight='bold')
        plt.xlabel("Metrics", fontsize=12)

        plt.grid(axis='y', linestyle='--', alpha=0.5)

        plt.xticks(rotation=15, fontsize=10)
        plt.yticks(fontsize=10)

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
        plt.savefig("ann_required_metrics_graph.png", dpi=300)
        plt.show()