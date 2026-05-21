from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# =========================
# 1. Загрузка и preprocessing
# =========================

def load_wine_data(red_path: str = "winequality-red.csv", white_path: str = "winequality-white.csv") -> pd.DataFrame:
    red = pd.read_csv(red_path, sep=";")
    white = pd.read_csv(white_path, sep=";")

    red["color"] = 0
    white["color"] = 1

    df = pd.concat([red, white], ignore_index=True)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def entropy(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def split_info(x: np.ndarray) -> float:
    _, counts = np.unique(x, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def info_gain(x: np.ndarray, y: np.ndarray) -> float:
    base = entropy(y)
    values, counts = np.unique(x, return_counts=True)
    conditional_entropy = 0.0

    for value, count in zip(values, counts):
        conditional_entropy += (count / len(y)) * entropy(y[x == value])

    return float(base - conditional_entropy)


def gain_ratio(x: np.ndarray, y: np.ndarray) -> float:
    si = split_info(x)
    if si == 0:
        return 0.0
    return info_gain(x, y) / si


def discretize_by_quantiles(s: pd.Series, bins: int = 4) -> pd.Series:
    return pd.qcut(s.rank(method="first"), q=bins, labels=False, duplicates="drop")


def preprocess_dataset(df: pd.DataFrame, corr_threshold: float = 0.85, top_features_count: int = 7) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    print("Исходный размер датасета после объединения и удаления дубликатов:", df.shape)
    print("\nПропуски по столбцам:")
    print(df.isna().sum())

    df = df.copy()
    df["quality_class"] = df["quality"].apply(lambda q: 0 if q <= 4 else (1 if q <= 6 else 2))

    corr = df.drop(columns=["quality_class"]).corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]

    print(f"\nСильно коррелирующие признаки, удаляемые при пороге > {corr_threshold}:")
    print(to_drop if to_drop else "Таких признаков не найдено")

    df = df.drop(columns=to_drop)

    os.makedirs("results_lab2", exist_ok=True)
    plt.figure(figsize=(11, 9))
    plt.imshow(df.drop(columns=["quality_class"]).corr(numeric_only=True), aspect="auto")
    plt.xticks(range(len(df.drop(columns=["quality_class"]).columns)), df.drop(columns=["quality_class"]).columns, rotation=90)
    plt.yticks(range(len(df.drop(columns=["quality_class"]).columns)), df.drop(columns=["quality_class"]).columns)
    plt.colorbar(label="correlation")
    plt.title("Корреляционная матрица после preprocessing")
    plt.tight_layout()
    plt.savefig("results_lab2/correlation_after_preprocessing.png", dpi=200)
    plt.close()

    X_for_rating = df.drop(columns=["quality", "quality_class"])
    y_class = df["quality_class"].to_numpy()
    X_disc = X_for_rating.apply(discretize_by_quantiles)

    scores = {col: gain_ratio(X_disc[col].to_numpy(), y_class) for col in X_disc.columns}
    rating = pd.Series(scores).sort_values(ascending=False)

    print("\nРейтинг признаков по Gain Ratio:")
    print(rating)

    selected_features = list(rating.head(top_features_count).index)
    print(f"\nВыбранные признаки для MLP-регрессора, top-{top_features_count}:")
    print(selected_features)

    return df, selected_features, rating


# =========================
# 2. Вспомогательные функции
# =========================

def train_test_split_manual(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(X))
    rng.shuffle(indices)

    test_count = int(len(X) * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


@dataclass
class StandardScalerManual:
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScalerManual":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler сначала нужно обучить через fit().")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# =========================
# 3. Самописный MLP-регрессор
# =========================

class MLPRegressorManual:
    def __init__(self, input_size: int, hidden1: int = 32, hidden2: int = 16, learning_rate: float = 0.01, random_state: int = 42):
        self.learning_rate = learning_rate
        rng = np.random.default_rng(random_state)

        # Инициализация Xavier/Glorot
        self.W1 = rng.normal(0, np.sqrt(2 / (input_size + hidden1)), size=(input_size, hidden1))
        self.b1 = np.zeros((1, hidden1))

        self.W2 = rng.normal(0, np.sqrt(2 / (hidden1 + hidden2)), size=(hidden1, hidden2))
        self.b2 = np.zeros((1, hidden2))

        self.W3 = rng.normal(0, np.sqrt(2 / (hidden2 + 1)), size=(hidden2, 1))
        self.b3 = np.zeros((1, 1))

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(a: np.ndarray) -> np.ndarray:
        return 1 - a ** 2

    def forward(self, X: np.ndarray):
        z1 = X @ self.W1 + self.b1
        a1 = self.tanh(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = self.tanh(z2)

        # Линейный выход: прогноз качества вина
        y_pred = a2 @ self.W3 + self.b3

        cache = (X, z1, a1, z2, a2, y_pred)
        return y_pred, cache

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    def backward(self, cache, y_true: np.ndarray):
        X, _z1, a1, _z2, a2, y_pred = cache
        n = X.shape[0]

        d_y_pred = 2 * (y_pred - y_true) / n

        # Выходной слой
        dW3 = a2.T @ d_y_pred
        db3 = np.sum(d_y_pred, axis=0, keepdims=True)

        # Ошибка уходит назад во 2 скрытый слой
        da2 = d_y_pred @ self.W3.T
        dz2 = da2 * self.tanh_derivative(a2)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Ошибка уходит назад в 1 скрытый слой
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.tanh_derivative(a1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2, dW3, db3

    def update_weights(self, gradients):
        dW1, db1, dW2, db2, dW3, db3 = gradients

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
            epochs: int = 500, batch_size: int = 64):
        history = {"train_mse": [], "val_mse": []}
        rng = np.random.default_rng(RANDOM_STATE)

        for epoch in range(1, epochs + 1):
            indices = np.arange(len(X_train))
            rng.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for start in range(0, len(X_train), batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred, cache = self.forward(X_batch)
                gradients = self.backward(cache, y_batch)
                self.update_weights(gradients)

            train_pred = self.predict(X_train)
            val_pred = self.predict(X_val)
            train_mse = self.mse(y_train, train_pred)
            val_mse = self.mse(y_val, val_pred)

            history["train_mse"].append(train_mse)
            history["val_mse"].append(val_mse)

            if epoch == 1 or epoch % 50 == 0:
                print(f"Эпоха {epoch:4d}: train MSE = {train_mse:.4f}, test MSE = {val_mse:.4f}")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred, _ = self.forward(X)
        return y_pred


# =========================
# 4. Метрики и запуск
# =========================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {"MSE": float(mse), "RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}


def save_plots(history: dict, y_true: np.ndarray, y_pred: np.ndarray):
    os.makedirs("results_lab2", exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(history["train_mse"], label="train MSE")
    plt.plot(history["val_mse"], label="test MSE")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE")
    plt.title("Изменение ошибки при обучении MLP")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results_lab2/loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.35)
    min_v = min(float(y_true.min()), float(y_pred.min()))
    max_v = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", label="идеальный прогноз")
    plt.xlabel("Реальное качество")
    plt.ylabel("Предсказанное качество")
    plt.title("MLP-регрессор: реальные и предсказанные значения")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results_lab2/predicted_vs_real.png", dpi=200)
    plt.close()


def main():
    df = load_wine_data()
    df, selected_features, rating = preprocess_dataset(df, corr_threshold=0.85, top_features_count=7)

    X = df[selected_features].to_numpy(dtype=float)
    y = df["quality"].to_numpy(dtype=float).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2, random_state=RANDOM_STATE)

    scaler_X = StandardScalerManual()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Масштабируем y, чтобы сети было проще обучаться, потом вернём прогнозы в исходную шкалу качества вина
    scaler_y = StandardScalerManual()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    model = MLPRegressorManual(
        input_size=X_train_scaled.shape[1],
        hidden1=32,
        hidden2=16,
        learning_rate=0.01,
        random_state=RANDOM_STATE,
    )

    print("\nНачинаем обучение MLP-регрессора...")
    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        X_test_scaled,
        y_test_scaled,
        epochs=500,
        batch_size=64,
    )

    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = y_pred_scaled * scaler_y.std_ + scaler_y.mean_

    metrics = regression_metrics(y_test, y_pred)
    print("\nИтоговые метрики на тестовой выборке:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nБиблиотечная реализация MLPRegressor из sklearn...")
    print("Начинаем обучение библиотечного MLPRegressor...")

    library_regressor = MLPRegressor(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        max_iter=1,
        warm_start=True,
        random_state=RANDOM_STATE,
    )

    library_epochs = 500
    library_history = {"train_mse": [], "val_mse": []}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        for epoch in range(1, library_epochs + 1):
            library_regressor.fit(X_train_scaled, y_train.ravel())

            library_train_pred = library_regressor.predict(X_train_scaled)
            library_test_pred = library_regressor.predict(X_test_scaled)

            library_train_mse = mean_squared_error(y_train.ravel(), library_train_pred)
            library_test_mse = mean_squared_error(y_test.ravel(), library_test_pred)

            library_history["train_mse"].append(library_train_mse)
            library_history["val_mse"].append(library_test_mse)

            if epoch == 1 or epoch % 50 == 0 or epoch == library_epochs:
                print(
                    f"Эпоха {epoch:4d}: "
                    f"train MSE = {library_train_mse:.4f}, "
                    f"test MSE = {library_test_mse:.4f}"
                )

    library_predictions = library_regressor.predict(X_test_scaled)
    library_mse = mean_squared_error(y_test.ravel(), library_predictions)
    library_rmse = np.sqrt(library_mse)
    library_mae = mean_absolute_error(y_test.ravel(), library_predictions)
    library_r2 = r2_score(y_test.ravel(), library_predictions)

    print("\nБиблиотечный MLPRegressor:")
    print(f"MSE : {library_mse:.4f}")
    print(f"RMSE: {library_rmse:.4f}")
    print(f"MAE : {library_mae:.4f}")
    print(f"R2  : {library_r2:.4f}")

    save_plots(history, y_test, y_pred)

    plt.figure(figsize=(9, 5))
    plt.plot(library_history["train_mse"], label="library train MSE")
    plt.plot(library_history["val_mse"], label="library test MSE")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE")
    plt.title("Изменение ошибки библиотечного MLPRegressor")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results_lab2/library_loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test.ravel(), library_predictions, alpha=0.35)
    min_v = min(float(y_test.min()), float(library_predictions.min()))
    max_v = max(float(y_test.max()), float(library_predictions.max()))
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", label="идеальный прогноз")
    plt.xlabel("Реальное качество")
    plt.ylabel("Предсказанное качество")
    plt.title("Библиотечный MLPRegressor: реальные и предсказанные значения")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results_lab2/library_predicted_vs_real.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
