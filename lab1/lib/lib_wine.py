import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    red = pd.read_csv('dataset/winequality-red.csv', sep=';')
    white = pd.read_csv('dataset/winequality-white.csv', sep=';')
    red['color'] = 0
    white['color'] = 1
    df = pd.concat([red, white], ignore_index=True).drop_duplicates().reset_index(drop=True)
    df['quality_class'] = df['quality'].apply(lambda q: 0 if q <= 4 else (1 if q <= 6 else 2))
    return df


def main():
    df = load_data()
    corr = df.drop(columns=['quality_class']).corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.85)]
    df = df.drop(columns=to_drop)
    print('Удалены сильно коррелирующие признаки:', to_drop)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.drop(columns=['quality_class']).corr(numeric_only=True), cmap='coolwarm')
    plt.title('Корреляционная матрица после очистки')
    plt.tight_layout()
    plt.show()

    X = df.drop(columns=['quality', 'quality_class'])
    y = df['quality_class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=50, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print('\nAccuracy:', round(accuracy_score(y_test, pred), 4))
    print('\nClassification report:\n', classification_report(y_test, pred))
    print('\nСтруктура дерева:\n')
    print(export_text(model, feature_names=list(X.columns), max_depth=5))

    importance = pd.DataFrame({'Признак': X.columns, 'Важность': model.feature_importances_})
    importance = importance.sort_values('Важность', ascending=False)
    print('\nТоп признаков по feature_importances_:')
    print(importance)


if __name__ == '__main__':
    main()
