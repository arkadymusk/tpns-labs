from __future__ import annotations

import gzip
import random
import struct
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


RANDOM_STATE = 42
DATA_DIR = Path("data")
RESULTS_DIR = Path("results_lab4_manual_cnn")


@dataclass
class ManualConfig:
    train_size: int = 15000
    test_size: int = 2500
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 0.001
    conv_filters: int = 4
    hidden_size: int = 32


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def find_mnist_file(filename: str) -> Path:
    """Ищет файл MNIST в папке data в распакованном или .gz варианте"""
    candidates = [
        DATA_DIR / "MNIST" / "raw" / filename,
        DATA_DIR / filename,
        DATA_DIR / "MNIST" / "raw" / f"{filename}.gz",
        DATA_DIR / f"{filename}.gz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Не найден файл {filename}. Сначала запустите библиотечную версию, "
        f"чтобы torchvision скачал MNIST в папку data."
    )


def read_idx_images(path: Path) -> np.ndarray:
    """Читает изображения MNIST из IDX-файла"""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"Некорректный IDX-файл изображений: {path}")
        data = np.frombuffer(file.read(), dtype=np.uint8)
    images = data.reshape(count, 1, rows, cols).astype(np.float32) / 255.0
    # Стандартная нормализация MNIST.
    images = (images - 0.1307) / 0.3081
    return images


def read_idx_labels(path: Path) -> np.ndarray:
    """Читает метки MNIST из IDX-файла"""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"Некорректный IDX-файл меток: {path}")
        labels = np.frombuffer(file.read(), dtype=np.uint8)
    return labels.astype(np.int64)


def load_mnist_subset(config: ManualConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Загружает подвыборку MNIST для ручной реализации"""
    train_images = read_idx_images(find_mnist_file("train-images-idx3-ubyte"))
    train_labels = read_idx_labels(find_mnist_file("train-labels-idx1-ubyte"))
    test_images = read_idx_images(find_mnist_file("t10k-images-idx3-ubyte"))
    test_labels = read_idx_labels(find_mnist_file("t10k-labels-idx1-ubyte"))

    rng = np.random.default_rng(RANDOM_STATE)
    train_idx = rng.choice(len(train_images), size=config.train_size, replace=False)
    test_idx = rng.choice(len(test_images), size=config.test_size, replace=False)

    return (
        train_images[train_idx],
        train_labels[train_idx],
        test_images[test_idx],
        test_labels[test_idx],
    )


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_backward(grad: np.ndarray, x: np.ndarray) -> np.ndarray:
    return grad * (x > 0)


def softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    """Возвращает loss и градиент по logits"""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    n = logits.shape[0]
    loss = -np.mean(np.log(probs[np.arange(n), labels] + 1e-12))

    grad = probs.copy()
    grad[np.arange(n), labels] -= 1.0
    grad /= n
    return loss, grad


def conv2d_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, tuple]:
    n, c, h, width = x.shape
    f, _, kh, kw = w.shape
    out_h = h - kh + 1
    out_w = width - kw + 1
    out = np.zeros((n, f, out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            window = x[:, :, i:i + kh, j:j + kw]
            # window: N,C,KH,KW; w: F,C,KH,KW -> N,F
            out[:, :, i, j] = np.tensordot(window, w, axes=([1, 2, 3], [1, 2, 3])) + b

    cache = (x, w, b)
    return out, cache


def conv2d_backward(grad_out: np.ndarray, cache: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ручной backward для conv2d_forward"""
    x, w, b = cache
    n, c, h, width = x.shape
    f, _, kh, kw = w.shape
    _, _, out_h, out_w = grad_out.shape

    grad_x = np.zeros_like(x)
    grad_w = np.zeros_like(w)
    grad_b = np.sum(grad_out, axis=(0, 2, 3))

    for i in range(out_h):
        for j in range(out_w):
            window = x[:, :, i:i + kh, j:j + kw]
            # grad_w: F,C,KH,KW
            grad_w += np.tensordot(grad_out[:, :, i, j], window, axes=([0], [0]))
            # grad_x window: N,C,KH,KW
            grad_x[:, :, i:i + kh, j:j + kw] += np.tensordot(
                grad_out[:, :, i, j], w, axes=([1], [0])
            )
    return grad_x, grad_w, grad_b


def avgpool2x2_forward(x: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Average pooling 2x2 со stride=2"""
    n, c, h, w = x.shape
    out_h = h // 2
    out_w = w // 2
    out = np.zeros((n, c, out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            window = x[:, :, 2 * i:2 * i + 2, 2 * j:2 * j + 2]
            out[:, :, i, j] = np.mean(window, axis=(2, 3))

    return out, (x.shape,)


def avgpool2x2_backward(grad_out: np.ndarray, cache: tuple) -> np.ndarray:
    """Backward для average pooling 2x2"""
    (x_shape,) = cache
    n, c, h, w = x_shape
    grad_x = np.zeros(x_shape, dtype=np.float32)
    out_h, out_w = grad_out.shape[2], grad_out.shape[3]

    for i in range(out_h):
        for j in range(out_w):
            grad = grad_out[:, :, i, j][:, :, None, None] / 4.0
            grad_x[:, :, 2 * i:2 * i + 2, 2 * j:2 * j + 2] += grad

    return grad_x


class ManualCNN:
    def __init__(self, config: ManualConfig) -> None:
        scale = 0.05
        self.w_conv = np.random.randn(config.conv_filters, 1, 3, 3).astype(np.float32) * scale
        self.b_conv = np.zeros(config.conv_filters, dtype=np.float32)

        flat_size = config.conv_filters * 13 * 13
        self.w1 = np.random.randn(flat_size, config.hidden_size).astype(np.float32) * scale
        self.b1 = np.zeros(config.hidden_size, dtype=np.float32)
        self.w2 = np.random.randn(config.hidden_size, 10).astype(np.float32) * scale
        self.b2 = np.zeros(10, dtype=np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        conv, conv_cache = conv2d_forward(x, self.w_conv, self.b_conv)  # N,4,26,26
        a_conv = relu(conv)
        pool, pool_cache = avgpool2x2_forward(a_conv)  # N,4,13,13
        flat = pool.reshape(x.shape[0], -1)
        z1 = flat @ self.w1 + self.b1
        a1 = relu(z1)
        logits = a1 @ self.w2 + self.b2
        cache = {
            "x": x,
            "conv": conv,
            "conv_cache": conv_cache,
            "a_conv": a_conv,
            "pool": pool,
            "pool_cache": pool_cache,
            "flat": flat,
            "z1": z1,
            "a1": a1,
        }
        return logits, cache

    def train_batch(self, x: np.ndarray, y: np.ndarray, lr: float) -> tuple[float, np.ndarray]:
        logits, cache = self.forward(x)
        loss, grad_logits = softmax_cross_entropy(logits, y)

        grad_w2 = cache["a1"].T @ grad_logits
        grad_b2 = np.sum(grad_logits, axis=0)
        grad_a1 = grad_logits @ self.w2.T
        grad_z1 = relu_backward(grad_a1, cache["z1"])

        grad_w1 = cache["flat"].T @ grad_z1
        grad_b1 = np.sum(grad_z1, axis=0)
        grad_flat = grad_z1 @ self.w1.T
        grad_pool = grad_flat.reshape(cache["pool"].shape)

        grad_a_conv = avgpool2x2_backward(grad_pool, cache["pool_cache"])
        grad_conv = relu_backward(grad_a_conv, cache["conv"])
        _, grad_w_conv, grad_b_conv = conv2d_backward(grad_conv, cache["conv_cache"])

        self.w2 -= lr * grad_w2
        self.b2 -= lr * grad_b2
        self.w1 -= lr * grad_w1
        self.b1 -= lr * grad_b1
        self.w_conv -= lr * grad_w_conv
        self.b_conv -= lr * grad_b_conv

        preds = np.argmax(logits, axis=1)
        return loss, preds

    def predict(self, x: np.ndarray, batch_size: int = 64) -> np.ndarray:
        preds: list[np.ndarray] = []
        for start in range(0, len(x), batch_size):
            batch = x[start:start + batch_size]
            logits, _ = self.forward(batch)
            preds.append(np.argmax(logits, axis=1))
        return np.concatenate(preds)


def iterate_batches(x: np.ndarray, y: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.random.permutation(len(x))
    for start in range(0, len(x), batch_size):
        batch_idx = indices[start:start + batch_size]
        yield x[batch_idx], y[batch_idx]


def evaluate_manual_model(
    model: ManualCNN, x: np.ndarray, y: np.ndarray, batch_size: int
) -> tuple[float, float, np.ndarray]:
    """Считает loss и accuracy без обновления весов, как evaluate в библиотечной версии"""
    losses: list[float] = []
    preds: list[np.ndarray] = []

    for start in range(0, len(x), batch_size):
        batch_x = x[start:start + batch_size]
        batch_y = y[start:start + batch_size]
        logits, _ = model.forward(batch_x)
        loss, _ = softmax_cross_entropy(logits, batch_y)
        losses.append(loss)
        preds.append(np.argmax(logits, axis=1))

    y_pred = np.concatenate(preds)
    return float(np.mean(losses)), accuracy_score(y, y_pred), y_pred


def plot_manual_curves(
    train_losses: list[float],
    test_losses: list[float],
    train_accs: list[float],
    test_accs: list[float],
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="manual train loss")
    plt.plot(test_losses, label="manual test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Softmax cross-entropy loss")
    plt.title("Ручная CNN: динамика ошибки")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "manual_loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label="manual train accuracy")
    plt.plot(test_accs, label="manual test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Ручная CNN: динамика accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "manual_accuracy_curve.png", dpi=200)
    plt.close()


def plot_manual_confusion_matrix(cm: np.ndarray) -> None:
    plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.title("Confusion matrix ручной CNN")
    plt.xlabel("Predicted digit")
    plt.ylabel("True digit")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.colorbar()
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "manual_confusion_matrix.png", dpi=200)
    plt.close()


def main() -> None:
    set_seed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = ManualConfig()

    print("Ручная NumPy-реализация CNN для MNIST")
    print(f"Используем выборку: train={config.train_size}, test={config.test_size}")

    x_train, y_train, x_test, y_test = load_mnist_subset(config)
    model = ManualCNN(config)

    train_losses: list[float] = []
    test_losses: list[float] = []
    train_accs: list[float] = []
    test_accs: list[float] = []

    print("Начинаем обучение ручной CNN...")
    for epoch in range(1, config.epochs + 1):
        batch_losses: list[float] = []
        batch_preds: list[np.ndarray] = []
        batch_targets: list[np.ndarray] = []

        for batch_x, batch_y in iterate_batches(x_train, y_train, config.batch_size):
            loss, preds = model.train_batch(batch_x, batch_y, config.learning_rate)
            batch_losses.append(loss)
            batch_preds.append(preds)
            batch_targets.append(batch_y)

        train_loss, train_acc, _ = evaluate_manual_model(
            model, x_train, y_train, config.batch_size
        )
        test_loss, test_acc, test_pred = evaluate_manual_model(
            model, x_test, y_test, config.batch_size
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

    final_loss, final_acc, final_pred = evaluate_manual_model(
        model, x_test, y_test, config.batch_size
    )
    cm = confusion_matrix(y_test, final_pred, labels=list(range(10)))

    print("\nИтоговая accuracy ручной CNN на уменьшенной test-выборке:")
    print(f"Accuracy: {final_acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, final_pred, digits=4, zero_division=0))
    print("Confusion matrix:")
    print(cm)

    plot_manual_curves(train_losses, test_losses, train_accs, test_accs)
    plot_manual_confusion_matrix(cm)

    report = f"""Лабораторная работа 4. Ручная реализация CNN для MNIST

Архитектура ручной модели:
- Conv2D вручную: 1 -> {config.conv_filters}, kernel_size=3
- ReLU
- Average Pooling 2x2 вручную
- Flatten: {config.conv_filters} * 13 * 13
- Fully Connected: {config.conv_filters * 13 * 13} -> {config.hidden_size}
- ReLU
- Fully Connected: {config.hidden_size} -> 10
- Softmax Cross-Entropy

Параметры:
- train_size: {config.train_size}
- test_size: {config.test_size}
- batch_size: {config.batch_size}
- epochs: {config.epochs}
- learning_rate: {config.learning_rate}

Итоговые результаты:
- train loss: {train_losses[-1]:.4f}
- test loss: {test_losses[-1]:.4f}
- train accuracy: {train_accs[-1]:.4f}
- test accuracy: {test_accs[-1]:.4f}

Classification report:
{classification_report(y_test, final_pred, digits=4, zero_division=0)}

Confusion matrix:
{cm}
"""
    (RESULTS_DIR / "manual_metrics_and_report.txt").write_text(report, encoding="utf-8")



if __name__ == "__main__":
    main()
