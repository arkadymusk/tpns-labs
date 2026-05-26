from __future__ import annotations

import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# =========================
# 1. Загрузка и preprocessing
# =========================

def load_wine_data(red_path: str = "dataset/winequality-red.csv", white_path: str = "dataset/winequality-white.csv") -> pd.DataFrame:
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

    os.makedirs("results_lab2_library_regressor", exist_ok=True)
    plt.figure(figsize=(11, 9))
    plt.imshow(df.drop(columns=["quality_class"]).corr(numeric_only=True), aspect="auto")
    plt.xticks(range(len(df.drop(columns=["quality_class"]).columns)), df.drop(columns=["quality_class"]).columns, rotation=90)
    plt.yticks(range(len(df.drop(columns=["quality_class"]).columns)), df.drop(columns=["quality_class"]).columns)
    plt.colorbar(label="correlation")
    plt.title("Корреляционная матрица после preprocessing")
    plt.tight_layout()
    plt.savefig("results_lab2_library_regressor/correlation_after_preprocessing.png", dpi=200)
    plt.close()

    X_for_rating = df.drop(columns=["quality", "quality_class"])
    y_class = df["quality_class"].to_numpy()
    X_disc = X_for_rating.apply(discretize_by_quantiles)

    scores = {col: gain_ratio(X_disc[col].to_numpy(), y_class) for col in X_disc.columns}
    rating = pd.Series(scores).sort_values(ascending=False)

    print("\nРейтинг признаков по Gain Ratio:")
    print(rating)

    selected_features = list(rating.head(top_features_count).index)
    print(f"\nВыбранные признаки для библиотечного MLP-регрессора, top-{top_features_count}:")
    print(selected_features)

    return df, selected_features, rating


# =========================
# 2. Графики и запуск
# =========================

def save_library_plots(history: dict, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    os.makedirs("results_lab2_library_regressor", exist_ok=True)

    epochs = range(1, len(history["train_mse"]) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["train_mse"], label="library train MSE")
    plt.plot(epochs, history["val_mse"], label="library test MSE")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE")
    plt.title("Изменение ошибки библиотечного MLPRegressor")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results_lab2_library_regressor/library_loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true.ravel(), y_pred, alpha=0.35)
    min_v = min(float(y_true.min()), float(y_pred.min()))
    max_v = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", label="идеальный прогноз")
    plt.xlabel("Реальное качество")
    plt.ylabel("Предсказанное качество")
    plt.title("Библиотечный MLPRegressor: реальные и предсказанные значения")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results_lab2_library_regressor/library_predicted_vs_real.png", dpi=200)
    plt.close()


def main() -> None:
    df = load_wine_data()
    df, selected_features, rating = preprocess_dataset(df, corr_threshold=0.85, top_features_count=7)

    X = df[selected_features].to_numpy(dtype=float)
    y = df["quality"].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

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
    history = {"train_mse": [], "val_mse": []}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        for epoch in range(1, library_epochs + 1):
            library_regressor.fit(X_train_scaled, y_train)

            train_pred = library_regressor.predict(X_train_scaled)
            test_pred = library_regressor.predict(X_test_scaled)

            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)

            history["train_mse"].append(train_mse)
            history["val_mse"].append(test_mse)

            if epoch == 1 or epoch % 50 == 0 or epoch == library_epochs:
                print(
                    f"Эпоха {epoch:4d}: "
                    f"train MSE = {train_mse:.4f}, "
                    f"test MSE = {test_mse:.4f}"
                )

    predictions = library_regressor.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nИтоговые метрики библиотечного MLPRegressor на тестовой выборке:")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R2  : {r2:.4f}")

    save_library_plots(history, y_test, predictions)
    print("\nГотово. Результаты сохранены в папку results_lab2_library_regressor")


if __name__ == "__main__":
    main()
