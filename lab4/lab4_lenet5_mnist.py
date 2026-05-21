from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


RANDOM_STATE = 42
RESULTS_DIR = Path("results_lab4_lenet5")
DATA_DIR = Path("data")


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 0.001
    test_size: float = 1 / 7  # примерно 60000 train и 10000 test при 70000 MNIST
    use_full_openml_mnist: bool = True


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LeNet5(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        return logits


def load_mnist_with_torchvision(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root=str(DATA_DIR), train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=str(DATA_DIR), train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader


def load_mnist_with_openml(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    from sklearn.datasets import fetch_openml

    print("torchvision недоступен, пробуем загрузить MNIST через sklearn OpenML...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)

    X = (X - 0.1307) / 0.3081
    X = X.reshape(-1, 1, 28, 28)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, test_loader


def load_mnist(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    try:
        return load_mnist_with_torchvision(config)
    except Exception as error:
        print(f"Не удалось загрузить MNIST через torchvision: {error}")
        return load_mnist_with_openml(config)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    all_preds: list[int] = []
    all_targets: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(labels.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_targets.extend(labels.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc, np.array(all_targets), np.array(all_preds)


def plot_loss(train_losses: list[float], test_losses: list[float]) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="train loss")
    plt.plot(test_losses, label="test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Динамика ошибки LeNet-5 на MNIST")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=200)
    plt.close()


def plot_accuracy(train_accs: list[float], test_accs: list[float]) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label="train accuracy")
    plt.plot(test_accs, label="test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Динамика accuracy LeNet-5 на MNIST")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_curve.png", dpi=200)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray) -> None:
    plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.title("Confusion matrix LeNet-5")
    plt.xlabel("Predicted digit")
    plt.ylabel("True digit")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.colorbar()

    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()


def plot_sample_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device, count: int = 16
) -> None:
    model.eval()
    images, labels = next(iter(loader))
    images_device = images.to(device)
    with torch.no_grad():
        outputs = model(images_device)
        preds = outputs.argmax(dim=1).cpu()

    shown = images[:count].clone()
    shown = shown * 0.3081 + 0.1307
    shown = shown.clamp(0, 1)

    plt.figure(figsize=(10, 10))
    for idx in range(count):
        plt.subplot(4, 4, idx + 1)
        plt.imshow(shown[idx].squeeze(0), cmap="gray")
        plt.title(f"real: {labels[idx].item()} / pred: {preds[idx].item()}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sample_predictions.png", dpi=200)
    plt.close()


def save_report(
    train_losses: list[float],
    test_losses: list[float],
    train_accs: list[float],
    test_accs: list[float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cm: np.ndarray,
    config: TrainConfig,
) -> None:
    report = classification_report(y_true, y_pred, digits=4)
    text = f"""Лабораторная работа 4. LeNet-5 для MNIST

Параметры обучения:
- batch_size: {config.batch_size}
- epochs: {config.epochs}
- learning_rate: {config.learning_rate}
- optimizer: Adam
- loss: CrossEntropyLoss

Итоговые результаты:
- train loss: {train_losses[-1]:.4f}
- test loss: {test_losses[-1]:.4f}
- train accuracy: {train_accs[-1]:.4f}
- test accuracy: {test_accs[-1]:.4f}

Classification report:
{report}

Confusion matrix:
{cm}
"""
    (RESULTS_DIR / "metrics_and_report.txt").write_text(text, encoding="utf-8")


def main() -> None:
    set_seed(RANDOM_STATE)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = TrainConfig()
    device = get_device()
    print(f"Используемое устройство: {device}")

    train_loader, test_loader = load_mnist(config)
    print(f"Размер train: {len(train_loader.dataset)}")
    print(f"Размер test : {len(test_loader.dataset)}")

    model = LeNet5().to(device)
    print("\nАрхитектура LeNet-5:")
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    train_losses: list[float] = []
    test_losses: list[float] = []
    train_accs: list[float] = []
    test_accs: list[float] = []

    print("\nНачинаем обучение LeNet-5...")
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc, y_true, y_pred = evaluate(
            model, test_loader, criterion, device
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(
            f"Epoch {epoch:2d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | test_loss={test_loss:.4f} | "
            f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f}"
        )

    final_loss, final_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))

    print("\nИтоговые результаты на тестовой выборке:")
    print(f"Test loss    : {final_loss:.4f}")
    print(f"Test accuracy: {final_acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix:")
    print(cm)

    plot_loss(train_losses, test_losses)
    plot_accuracy(train_accs, test_accs)
    plot_confusion_matrix(cm)
    plot_sample_predictions(model, test_loader, device)
    save_report(train_losses, test_losses, train_accs, test_accs, y_true, y_pred, cm, config)

    torch.save(model.state_dict(), RESULTS_DIR / "lenet5_mnist_state_dict.pt")


if __name__ == "__main__":
    main()
