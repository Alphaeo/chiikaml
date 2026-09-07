#pragma once

#include <vector>

#include "chiikaml/matrix.hpp"


namespace chiikaml::metrics {

// Classification metrics for evaluating model performance.

// Accuracy : the proportion of correct predictions among the total number of predictions.
double accuracy(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred
);

// Precision : the proportion of true positive predictions among all positive predictions made by the model.
double precision(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred
);

// Recall : the proportion of true positive predictions among all actual positive instances in the dataset.
double recall(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred
);

// F1 Score : the harmonic mean of precision and recall, providing a single metric that balances both aspects of model performance.
double f1_score(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred
);

// Currently only supports binary classification with labels 0 and 1, where 1 is considered the positive class.
// TODO: Extend support to multi-class classification by computing precision for each class and returning a vector of precision values.

// Confusion Matrix : a square matrix that summarizes the performance of a classification model
// by showing the counts of true positive, true negative, false positive, and false negative predictions for each class.
Matrix confusion_matrix(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred
);

}