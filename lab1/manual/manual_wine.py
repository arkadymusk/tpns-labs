import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data():
    red = pd.read_csv('dataset/winequality-red.csv', sep=';')
    white = pd.read_csv('dataset/winequality-white.csv', sep=';')
    red['color'] = 0
    white['color'] = 1
    df = pd.concat([red, white], ignore_index=True)
    df['quality_class'] = df['quality'].apply(lambda q: 0 if q <= 4 else (1 if q <= 6 else 2))
    return df.drop_duplicates().reset_index(drop=True)


def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p + 1e-12))


def split_info(x):
    _, counts = np.unique(x, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p + 1e-12))


def info_gain(x, y):
    base = entropy(y)
    values, counts = np.unique(x, return_counts=True)
    cond = sum((c / len(y)) * entropy(y[x == v]) for v, c in zip(values, counts))
    return base - cond


def gain_ratio(x, y):
    si = split_info(x)
    return info_gain(x, y) / si if si > 0 else 0


def discretize_by_quantiles(s, bins=4):
    return pd.qcut(s.rank(method='first'), q=bins, labels=False)


def main():
    df = load_data()
    print('Размер датасета после объединения и удаления дубликатов:', df.shape)
    print('Пропуски по столбцам:\n', df.isna().sum())

    corr = df.drop(columns=['quality_class']).corr(numeric_only=True).abs()

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.85)]
    print('\nУдаляем сильно коррелирующие признаки (>|0.85|):', to_drop)

    df = df.drop(columns=to_drop)
    corr2 = df.drop(columns=['quality_class']).corr(numeric_only=True).abs()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(corr, cmap='coolwarm', ax=axes[0])
    axes[0].set_title('Корреляционная матрица ДО удаления признаков')

    sns.heatmap(corr2, cmap='coolwarm', ax=axes[1])
    axes[1].set_title('Корреляционная матрица ПОСЛЕ удаления признаков')

    plt.tight_layout()
    plt.show()

    X = df.drop(columns=['quality', 'quality_class'])
    y = df['quality_class'].to_numpy()
    X_disc = X.apply(discretize_by_quantiles)

    scores = {col: gain_ratio(X_disc[col].to_numpy(), y) for col in X_disc.columns}
    rating = pd.Series(scores).sort_values(ascending=False)
    print('\nТоп признаков по Gain Ratio (C4.5):')
    print(rating)
    print('\nЛучшие признаки для настройки модели:', list(rating.head(5).index))


if __name__ == '__main__':
    main()