"""
Run a KNN Grid Search on the Wisconsin breast-cancer dataset.

Usage:

    $env:PYTHONPATH = "build-python/python"
    python python/benchmarks/evaluate_models_gridsearch.py
"""

import os

if os.name == "nt":
    mingw_bin = r"C:\msys64\ucrt64\bin"

    if os.path.isdir(mingw_bin):
        os.add_dll_directory(mingw_bin)

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

import chiikaml


def to_chiikaml_matrix(values):
    matrix = chiikaml.Matrix(
        values.shape[0],
        values.shape[1],
    )

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            matrix[i, j] = float(values[i, j])

    return matrix


def main():
    dataset = load_breast_cancer()

    X_numpy = np.asarray(
        dataset.data,
        dtype=np.float64,
    )

    # Reverse labels so that:
    #
    #     1 = malignant
    #     0 = benign
    y = (
        1 - np.asarray(
            dataset.target,
            dtype=np.int32,
        )
    ).tolist()

    X = to_chiikaml_matrix(X_numpy)

    split = chiikaml.model_selection.train_test_split(
        X,
        y,
        train_size=0.8,
        shuffle=True,
        seed=42,
    )

    # Convert the split matrices back to NumPy for StandardScaler.
    X_train_numpy = np.array([
        [
            split.X_train[i, j]
            for j in range(split.X_train.cols())
        ]
        for i in range(split.X_train.rows())
    ])

    X_test_numpy = np.array([
        [
            split.X_test[i, j]
            for j in range(split.X_test.cols())
        ]
        for i in range(split.X_test.rows())
    ])

    scaler = StandardScaler()

    X_train_numpy = scaler.fit_transform(
        X_train_numpy
    )

    X_test_numpy = scaler.transform(
        X_test_numpy
    )

    X_train = to_chiikaml_matrix(X_train_numpy)
    X_test = to_chiikaml_matrix(X_test_numpy)

    search = chiikaml.model_selection.grid_search_knn(
        X_train,
        split.y_train,
        k_values=[1, 3, 5, 7, 9, 11, 13, 15],
        n_splits=5,
        shuffle=True,
        seed=42,
        metric="f1",
    )

    print("=== Grid Search results ===")

    for candidate in search["candidate_results"]:
        print(
            f"k={candidate['k']:2d} | "
            f"mean F1={candidate['mean_score']:.4f} | "
            f"std={candidate['standard_deviation']:.4f}"
        )

    print()
    print(f"Best k: {search['best_k']}")
    print(f"Best cross-validation F1: {search['best_score']:.4f}")

    best_model = search["best_model"]
    predictions = best_model.predict(X_test)

    print()
    print("=== Test results ===")
    print(
        "Accuracy:",
        chiikaml.metrics.accuracy(
            split.y_test,
            predictions,
        ),
    )
    print(
        "Precision:",
        chiikaml.metrics.precision(
            split.y_test,
            predictions,
        ),
    )
    print(
        "Recall:",
        chiikaml.metrics.recall(
            split.y_test,
            predictions,
        ),
    )
    print(
        "F1 score:",
        chiikaml.metrics.f1_score(
            split.y_test,
            predictions,
        ),
    )
    print("Confusion matrix:")
    print(
        chiikaml.metrics.confusion_matrix(
            split.y_test,
            predictions,
        )
    )


if __name__ == "__main__":
    main()