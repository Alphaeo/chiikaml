#pragma once

#include <functional>
#include <vector>

#include "chiikaml/matrix.hpp"
#include "chiikaml/model_selection/k_fold.hpp"

namespace chiikaml::model_selection {

// Contains the validation score obtained for each fold, followed by
// their mean and standard deviation.
struct CrossValidationResult {
    std::vector<double> fold_scores;
    double mean_score;
    double standard_deviation;
};

// Function used to compare the true labels with the predictions.
using ClassificationMetric = std::function<
    double(
        const std::vector<int>& y_true,
        const std::vector<int>& y_pred
    )
>;

// Evaluates a classification model using K-fold cross-validation.
//
// The supplied model is copied for every fold. Each copy is fitted
// independently, preventing fitted state from leaking between folds.
//
// Model must provide:
//
//     void fit(const Matrix&, const std::vector<int>&);
//     std::vector<int> predict(const Matrix&) const;
template <typename Model>
CrossValidationResult cross_validate_classification(
    const Model& model,
    const Matrix& X,
    const std::vector<int>& y,
    const KFold& splitter,
    const ClassificationMetric& metric
);

} // namespace chiikaml::model_selection

#include "chiikaml/model_selection/cross_validation.tpp"