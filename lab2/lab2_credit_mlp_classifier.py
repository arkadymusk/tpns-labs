from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
RESULTS_DIR = "results_lab2_classifier"


# =========================================================
# 1. Загрузка данных
# =========================================================

def load_credit_data() -> pd.DataFrame:
    possible_paths = [
        "dataset/default_of_credit_card_clients.csv",
        "default of credit card clients.csv",
        "default of credit card clients.xls",
    ]

    existing_path = None
    for path in possible_paths:
        if os.path.exists(path):
            existing_path = path
            break

    if existing_path is None:
        raise FileNotFoundError(
            "Не найден файл датасета. Положите рядом со скриптом файл "
            "default_of_credit_card_clients.csv или default of credit card clients.xls"
        )

    if existing_path.endswith(".csv"):
        df = pd.read_csv(existing_path)
    else:
        df = pd.read_excel(existing_path, header=1)

    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed") or str(c).strip() == ""]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    first_row = df.iloc[0].astype(str).str.lower().tolist()
    if "limit_bal" in first_row or "default payment next month" in first_row:
        df = df.iloc[1:].reset_index(drop=True)

    rename_map = {
        "X1": "LIMIT_BAL",
        "X2": "SEX",
        "X3": "EDUCATION",
        "X4": "MARRIAGE",
        "X5": "AGE",
        "X6": "PAY_0",
        "X7": "PAY_2",
        "X8": "PAY_3",
        "X9": "PAY_4",
        "X10": "PAY_5",
        "X11": "PAY_6",
        "X12": "BILL_AMT1",
        "X13": "BILL_AMT2",
        "X14": "BILL_AMT3",
        "X15": "BILL_AMT4",
        "X16": "BILL_AMT5",
        "X17": "BILL_AMT6",
        "X18": "PAY_AMT1",
        "X19": "PAY_AMT2",
        "X20": "PAY_AMT3",
        "X21": "PAY_AMT4",
        "X22": "PAY_AMT5",
        "X23": "PAY_AMT6",
        "Y": "DEFAULT_NEXT_MONTH",
        "default payment next month": "DEFAULT_NEXT_MONTH",
    }
    df = df.rename(columns=rename_map)

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =========================================================
# 2. Gain Ratio из C4.5 для отбора признаков
# =========================================================

def entropy(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def split_info(x: np.ndarray) -> float:
    _, counts = np.unique(x, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def info_gain(x: np.ndarray, y: np.ndarray) -> float:
    base_entropy = entropy(y)
    values, counts = np.unique(x, return_counts=True)
    conditional_entropy = 0.0

    for value, count in zip(values, counts):
        conditional_entropy += (count / len(y)) * entropy(y[x == value])

    return float(base_entropy - conditional_entropy)


def gain_ratio(x: np.ndarray, y: np.ndarray) -> float:
    si = split_info(x)
    if si == 0:
        return 0.0
    return info_gain(x, y) / si


def discretize_by_quantiles(s: pd.Series, bins: int = 5) -> pd.Series:
    if s.nunique() <= bins:
        return s.astype(int)
    return pd.qcut(s.rank(method="first"), q=bins, labels=False, duplicates="drop")


# =========================================================
# 3. Preprocessing по логике первой лабораторной
# =========================================================

def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "EDUCATION" in df.columns:
        df.loc[~df["EDUCATION"].isin([1, 2, 3, 4]), "EDUCATION"] = 4

    if "MARRIAGE" in df.columns:
        df.loc[~df["MARRIAGE"].isin([1, 2, 3]), "MARRIAGE"] = 3

    if "SEX" in df.columns:
        mode = int(df["SEX"].mode().iloc[0])
        df.loc[~df["SEX"].isin([1, 2]), "SEX"] = mode

    return df


def preprocess_dataset(
    df: pd.DataFrame,
    corr_threshold: float = 0.90,
    top_features_count: int = 15,
) -> Tuple[pd.DataFrame, List[str], pd.Series, List[str]]:
    print("Исходный размер датасета:", df.shape)
    print("\nПропуски до обработки:")
    print(df.isna().sum())

    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    df = clean_categories(df)

    target_col = "DEFAULT_NEXT_MONTH"
    if target_col not in df.columns:
        raise ValueError(f"Не найден целевой столбец {target_col}. Найденные столбцы: {list(df.columns)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_all = df.drop(columns=[target_col])
    corr = X_all.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    correlated_to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]

    print(f"\nСильно коррелирующие признаки, удаляемые при пороге > {corr_threshold}:")
    print(correlated_to_drop if correlated_to_drop else "Таких признаков не найдено")

    X_after_corr = X_all.drop(columns=correlated_to_drop)

    # Сохраняем корреляционную матрицу после удаления зависимых признаков
    corr_after = X_after_corr.corr(numeric_only=True)
    plt.figure(figsize=(13, 10))
    plt.imshow(corr_after, aspect="auto")
    plt.xticks(range(len(corr_after.columns)), corr_after.columns, rotation=90)
    plt.yticks(range(len(corr_after.columns)), corr_after.columns)
    plt.colorbar(label="correlation")
    plt.title("Корреляционная матрица после preprocessing")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "correlation_after_preprocessing.png"), dpi=200)
    plt.close()

    # Gain Ratio для отбора признаков
    y = df[target_col].astype(int).to_numpy()
    X_disc = X_after_corr.apply(discretize_by_quantiles)
    scores: Dict[str, float] = {col: gain_ratio(X_disc[col].to_numpy(), y) for col in X_disc.columns}
    rating = pd.Series(scores).sort_values(ascending=False)

    print("\nРейтинг признаков по Gain Ratio:")
    print(rating)

    selected_features = list(rating.head(top_features_count).index)
    print(f"\nВыбранные признаки для MLP-классификатора, top-{top_features_count}:")
    print(selected_features)

    # График Gain Ratio
    plt.figure(figsize=(12, 6))
    rating.sort_values().plot(kind="barh")
    plt.title("Оценка признаков по Gain Ratio")
    plt.xlabel("Gain Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "gain_ratio_features.png"), dpi=200)
    plt.close()

    return df, selected_features, rating, correlated_to_drop


# =========================================================
# 4. Вспомогательные функции
# =========================================================

def train_test_split_stratified(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    train_indices: List[int] = []
    test_indices: List[int] = []

    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        test_count = int(len(cls_indices) * test_size)
        test_indices.extend(cls_indices[:test_count])
        train_indices.extend(cls_indices[test_count:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


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


# =========================================================
# 5. Самописный MLP-классификатор
# =========================================================

class MLPBinaryClassifierManual:
    def __init__(
        self,
        input_size: int,
        hidden1: int = 32,
        hidden2: int = 16,
        learning_rate: float = 0.01,
        random_state: int = 42,
    ):
        self.learning_rate = learning_rate
        rng = np.random.default_rng(random_state)

        self.W1 = rng.normal(0, np.sqrt(2 / input_size), size=(input_size, hidden1))
        self.b1 = np.zeros((1, hidden1))

        self.W2 = rng.normal(0, np.sqrt(2 / hidden1), size=(hidden1, hidden2))
        self.b2 = np.zeros((1, hidden2))

        self.W3 = rng.normal(0, np.sqrt(2 / hidden2), size=(hidden2, 1))
        self.b3 = np.zeros((1, 1))

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)

        self.z3 = self.a2 @ self.W3 + self.b3
        self.y_prob = self.sigmoid(self.z3)
        return self.y_prob

    def backward(self, X: np.ndarray, y_true: np.ndarray) -> None:
        m = X.shape[0]
        y_true = y_true.reshape(-1, 1)

        dz3 = (self.y_prob - y_true) / m
        dW3 = self.a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * self.relu_derivative(self.z2)
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 200,
        batch_size: int = 128,
        verbose_every: int = 20,
    ) -> Dict[str, List[float]]:
        history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
        rng = np.random.default_rng(RANDOM_STATE)

        for epoch in range(1, epochs + 1):
            indices = np.arange(len(X_train))
            rng.shuffle(indices)

            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                self.forward(X_batch)
                self.backward(X_batch, y_batch)

            train_prob = self.forward(X_train)
            val_prob = self.forward(X_val)

            train_loss = self.binary_cross_entropy(y_train.reshape(-1, 1), train_prob)
            val_loss = self.binary_cross_entropy(y_val.reshape(-1, 1), val_prob)
            train_acc = accuracy_score_manual(y_train, (train_prob >= 0.5).astype(int).ravel())
            val_acc = accuracy_score_manual(y_val, (val_prob >= 0.5).astype(int).ravel())

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)

            if epoch == 1 or epoch % verbose_every == 0 or epoch == epochs:
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}"
                )

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).ravel()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# =========================================================
# 6. Метрики классификации вручную
# =========================================================

def confusion_matrix_manual(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def accuracy_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def precision_recall_f1_manual(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    cm = confusion_matrix_manual(y_true, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0
    return float(precision), float(recall), float(f1)


def plot_confusion_matrix(cm: np.ndarray, path: str) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion matrix")
    plt.xticks([0, 1], ["pred 0", "pred 1"])
    plt.yticks([0, 1], ["true 0", "true 1"])
    plt.colorbar()

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# =========================================================
# 7. Main
# =========================================================

def main() -> None:
    df_raw = load_credit_data()
    df, selected_features, gain_rating, dropped_corr = preprocess_dataset(
        df_raw,
        corr_threshold=0.90,
        top_features_count=15,
    )

    target_col = "DEFAULT_NEXT_MONTH"
    X = df[selected_features].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=int)

    print("\nРаспределение классов:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"class {cls}: {cnt} ({cnt / len(y):.2%})")

    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.2, random_state=RANDOM_STATE)

    scaler = StandardScalerManual()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPBinaryClassifierManual(
        input_size=X_train_scaled.shape[1],
        hidden1=32,
        hidden2=16,
        learning_rate=0.01,
        random_state=RANDOM_STATE,
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        epochs=50,
        batch_size=128,
        verbose_every=10,
    )

    y_prob = model.predict_proba(X_test_scaled)
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix_manual(y_test, y_pred)
    accuracy = accuracy_score_manual(y_test, y_pred)
    precision, recall, f1 = precision_recall_f1_manual(y_test, y_pred)

    print("\nИтоговые метрики на test:")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nБиблиотечная реализация MLPClassifier из sklearn...")
    print("Начинаем обучение библиотечного MLPClassifier...")

    library_classifier = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=1,
        warm_start=True,
        random_state=RANDOM_STATE
    )

    library_epochs = 50
    library_history = {
        "train_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "test_accuracy": [],
    }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        for epoch in range(1, library_epochs + 1):
            library_classifier.fit(X_train_scaled, y_train)

            train_proba = library_classifier.predict_proba(X_train_scaled)
            test_proba = library_classifier.predict_proba(X_test_scaled)
            train_pred = library_classifier.predict(X_train_scaled)
            test_pred = library_classifier.predict(X_test_scaled)

            train_loss = log_loss(y_train, train_proba)
            test_loss = log_loss(y_test, test_proba)
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)

            library_history["train_loss"].append(train_loss)
            library_history["test_loss"].append(test_loss)
            library_history["train_accuracy"].append(train_acc)
            library_history["test_accuracy"].append(test_acc)

            if epoch == 1 or epoch % 10 == 0 or epoch == library_epochs:
                print(
                    f"Epoch {epoch:4d}/{library_epochs} | "
                    f"train_loss={train_loss:.4f} | test_loss={test_loss:.4f} | "
                    f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f}"
                )

    library_pred = library_classifier.predict(X_test_scaled)
    library_cm = confusion_matrix(y_test, library_pred)
    library_accuracy = accuracy_score(y_test, library_pred)
    library_precision = precision_score(y_test, library_pred, zero_division=0)
    library_recall = recall_score(y_test, library_pred, zero_division=0)
    library_f1 = f1_score(y_test, library_pred, zero_division=0)

    print("\nИтоговые метрики библиотечного MLPClassifier на test:")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(library_cm)
    print(f"Accuracy : {library_accuracy:.4f}")
    print(f"Precision: {library_precision:.4f}")
    print(f"Recall   : {library_recall:.4f}")
    print(f"F1-score : {library_f1:.4f}")

    # Матрица ошибок для библиотечной реализации
    plot_confusion_matrix(
        library_cm,
        os.path.join(RESULTS_DIR, "library_confusion_matrix.png")
    )

    # График функции ошибки для библиотечной реализации
    plt.figure(figsize=(9, 5))
    plt.plot(library_history["train_loss"], label="library train loss")
    plt.plot(library_history["test_loss"], label="library test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Log loss")
    plt.title("Динамика ошибки библиотечного MLPClassifier")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "library_loss_curve.png"), dpi=200)
    plt.close()

    # График accuracy для библиотечной реализации
    plt.figure(figsize=(9, 5))
    plt.plot(library_history["train_accuracy"], label="library train accuracy")
    plt.plot(library_history["test_accuracy"], label="library test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Динамика accuracy библиотечного MLPClassifier")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "library_accuracy_curve.png"), dpi=200)
    plt.close()

    # График функции ошибки
    plt.figure(figsize=(9, 5))
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.title("Динамика ошибки MLP-классификатора")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"), dpi=200)
    plt.close()

    # График accuracy
    plt.figure(figsize=(9, 5))
    plt.plot(history["train_accuracy"], label="train accuracy")
    plt.plot(history["val_accuracy"], label="test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Динамика accuracy MLP-классификатора")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_curve.png"), dpi=200)
    plt.close()

    plot_confusion_matrix(cm, os.path.join(RESULTS_DIR, "confusion_matrix.png"))


if __name__ == "__main__":
    main()
