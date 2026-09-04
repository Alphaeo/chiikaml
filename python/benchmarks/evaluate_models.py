"""
Evaluate chiikaml classification models on the Wisconsin breast
cancer dataset bundled with scikit-learn.

The script:

1. loads a real binary-classification dataset;
2. splits it using chiikaml.model_selection.train_test_split;
3. standardizes features using training-set statistics only;
4. fits several chiikaml classifiers;
5. evaluates them with chiikaml metrics.

Usage from the repository root:

    $env:PYTHONPATH = "build-python/python"
    python python/evaluate_models.py
"""

import os
import time

# Must be done before importing the native chiikaml module.
if os.name == "nt":
    mingw_bin = r"C:\msys64\ucrt64\bin"

    if os.path.isdir(mingw_bin):
        os.add_dll_directory(mingw_bin)

import numpy as np
from sklearn.datasets import load_breast_cancer

import chiikaml


def to_chiikaml_matrix(values):
    """Convert a NumPy array to a chiikaml Matrix."""

    matrix = chiikaml.Matrix(
        values.shape[0],
        values.shape[1],
    )

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            matrix[i, j] = float(values[i, j])

    return matrix


def standardize_train_test(X_train, X_test):
    """
    Standardize features using only the training-set statistics.

    Fitting the transformation exclusively on training data avoids
    leaking information from the test set.
    """

    n_train = X_train.rows()
    n_test = X_test.rows()
    n_features = X_train.cols()

    means = [0.0] * n_features
    standard_deviations = [0.0] * n_features

    for j in range(n_features):
        for i in range(n_train):
            means[j] += X_train[i, j]

        means[j] /= n_train

    for j in range(n_features):
        for i in range(n_train):
            difference = X_train[i, j] - means[j]
            standard_deviations[j] += difference * difference

        standard_deviations[j] = (
            standard_deviations[j] / n_train
        ) ** 0.5

        # A constant feature has no scale and can simply use 1.
        if standard_deviations[j] == 0.0:
            standard_deviations[j] = 1.0

    standardized_train = chiikaml.Matrix(
        n_train,
        n_features,
    )

    standardized_test = chiikaml.Matrix(
        n_test,
        n_features,
    )

    for i in range(n_train):
        for j in range(n_features):
            standardized_train[i, j] = (
                X_train[i, j] - means[j]
            ) / standard_deviations[j]

    for i in range(n_test):
        for j in range(n_features):
            standardized_test[i, j] = (
                X_test[i, j] - means[j]
            ) / standard_deviations[j]

    return standardized_train, standardized_test


def time_call(function, *args):
    """Execute a function and return its result and execution time."""

    start = time.perf_counter()
    result = function(*args)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return result, elapsed_ms


def encode_svm_labels(labels):
    """
    Convert binary labels from 0/1 to -1/+1.

    BinarySVM is an internal binary solver and therefore works with
    signed labels.
    """

    return [
        1 if label == 1 else -1
        for label in labels
    ]


def decode_svm_predictions(predictions):
    """Convert BinarySVM predictions from -1/+1 back to 0/1."""

    return [
        1 if prediction == 1 else 0
        for prediction in predictions
    ]


def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    uses_signed_labels=False,
):
    """
    Fit and evaluate one classification model.

    uses_signed_labels must be true for BinarySVM because its internal
    labels are encoded as -1 and +1.
    """

    training_labels = y_train

    if uses_signed_labels:
        training_labels = encode_svm_labels(y_train)

    _, fit_time = time_call(
        model.fit,
        X_train,
        training_labels,
    )

    predictions, predict_time = time_call(
        model.predict,
        X_test,
    )

    if uses_signed_labels:
        predictions = decode_svm_predictions(predictions)

    accuracy = chiikaml.metrics.accuracy(
        y_test,
        predictions,
    )

    precision = chiikaml.metrics.precision(
        y_test,
        predictions,
    )

    recall = chiikaml.metrics.recall(
        y_test,
        predictions,
    )

    f1 = chiikaml.metrics.f1_score(
        y_test,
        predictions,
    )

    confusion = chiikaml.metrics.confusion_matrix(
        y_test,
        predictions,
    )

    print(f"=== {name} ===")
    print(f"Fit time:     {fit_time:8.3f} ms")
    print(f"Predict time: {predict_time:8.3f} ms")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"F1 score:     {f1:.4f}")
    print("Confusion matrix:")
    print(confusion)

    if uses_signed_labels:
        print(
            "Support vectors: "
            f"{model.number_of_support_vectors()}"
        )
        print(f"Converged:       {model.converged()}")
        print(f"SMO iterations:  {model.iterations()}")
        print(f"Effective gamma: {model.gamma():.6f}")

    print()

    return {
        "name": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fit_time": fit_time,
        "predict_time": predict_time,
    }


def main():
    dataset = load_breast_cancer()

    X_numpy = np.asarray(
        dataset.data,
        dtype=np.float64,
    )

    # In the original sklearn dataset:
    #
    #     0 = malignant
    #     1 = benign
    #
    # The current chiikaml precision, recall and F1 implementations
    # treat label 1 as positive. Reverse the labels so that malignant
    # tumors are the positive class.
    y_numpy = 1 - np.asarray(
        dataset.target,
        dtype=np.int32,
    )

    X = to_chiikaml_matrix(X_numpy)
    y = y_numpy.tolist()

    split = chiikaml.model_selection.train_test_split(
        X,
        y,
        train_size=0.8,
        shuffle=True,
        seed=42,
    )

    X_train, X_test = standardize_train_test(
        split.X_train,
        split.X_test,
    )

    print("Wisconsin Breast Cancer dataset")
    print(f"Training samples: {X_train.rows()}")
    print(f"Test samples:     {X_test.rows()}")
    print(f"Features:         {X_train.cols()}")
    print("Positive label:   1 = malignant")
    print()

    # Each entry contains:
    #
    #     model name
    #     model instance
    #     whether the model expects labels encoded as -1/+1
    models = [
        (
            "KNNClassifier",
            chiikaml.KNNClassifier(5),
            False,
        ),
        (
            "DecisionTreeClassifier",
            chiikaml.DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=2,
            ),
            False,
        ),
        (
            "RandomForestClassifier",
            chiikaml.RandomForestClassifier(
                n_trees=50,
                max_depth=7,
                min_samples_split=2,
                seed=42,
            ),
            False,
        ),
        (
            "BinarySVM (linear)",
            chiikaml.BinarySVM(
                C=1.0,
                kernel=chiikaml.SVMKernel.Linear,
                gamma=0.0,
                max_iterations=5000,
                tolerance=1e-4,
                fit_intercept=True,
                seed=42,
            ),
            True,
        ),
        (
            "BinarySVM (RBF)",
            chiikaml.BinarySVM(
                C=1.0,
                kernel=chiikaml.SVMKernel.RBF,
                gamma=0.0,
                max_iterations=5000,
                tolerance=1e-4,
                fit_intercept=True,
                seed=42,
            ),
            True,
        ),
    ]

    results = []

    for name, model, uses_signed_labels in models:
        result = evaluate_model(
            name,
            model,
            X_train,
            split.y_train,
            X_test,
            split.y_test,
            uses_signed_labels=uses_signed_labels,
        )

        results.append(result)

    print("=== Summary ===")

    results.sort(
        key=lambda result: result["f1"],
        reverse=True,
    )

    print(
        f"{'Model':28}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
    )

    for result in results:
        print(
            f"{result['name']:28}"
            f"{result['accuracy']:12.4f}"
            f"{result['precision']:12.4f}"
            f"{result['recall']:12.4f}"
            f"{result['f1']:12.4f}"
        )


if __name__ == "__main__":
    main()