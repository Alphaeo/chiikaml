#include "chiikaml/model_selection/cross_validation.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace chiikaml::model_selection {

namespace {

// Selects vector elements in the order specified by indices.
//
// This helper is private to this implementation because the public
// cross-validation interface does not need to expose it.
std::vector<int> select_elements(
    const std::vector<int>& values,
    const std::vector<std::size_t>& indices
) {
    std::vector<int> result;
    result.reserve(indices.size());

    for (std::size_t index : indices) {
        if (index >= values.size()) {
            throw std::out_of_range(
                "Target index is outside the vector"
            );
        }

        result.push_back(values[index]);
    }

    return result;
}

} // namespace

CrossValidationResult cross_validate_classification(
    const Matrix& X,
    const std::vector<int>& y,
    const KFold& splitter,
    const ClassificationEvaluator& evaluator,
    const ClassificationMetric& metric
) {
    if (X.rows() != y.size()) {
        throw std::invalid_argument(
            "X.rows() must be equal to y.size()"
        );
    }

    if (X.rows() == 0) {
        throw std::invalid_argument(
            "The dataset must not be empty"
        );
    }

    if (!evaluator) {
        throw std::invalid_argument(
            "The model evaluator must be valid"
        );
    }

    if (!metric) {
        throw std::invalid_argument(
            "The metric function must be valid"
        );
    }

    const std::vector<FoldIndices> folds =
        splitter.split(X.rows());

    CrossValidationResult result;
    result.fold_scores.reserve(folds.size());

    for (const FoldIndices& fold : folds) {
        const std::vector<std::size_t>& training_indices =
            fold.first;

        const std::vector<std::size_t>& validation_indices =
            fold.second;

        Matrix X_train =
            X.select_rows(training_indices);

        Matrix X_validation =
            X.select_rows(validation_indices);

        std::vector<int> y_train =
            select_elements(y, training_indices);

        std::vector<int> y_validation =
            select_elements(y, validation_indices);

        // The evaluator is responsible for constructing and fitting
        // a fresh model for this fold.
        const std::vector<int> predictions = evaluator(
            X_train,
            y_train,
            X_validation
        );

        if (predictions.size() != y_validation.size()) {
            throw std::runtime_error(
                "The model returned an incorrect number of predictions"
            );
        }

        result.fold_scores.push_back(
            metric(y_validation, predictions)
        );
    }

    double score_sum = 0.0;

    for (double score : result.fold_scores) {
        if (!std::isfinite(score)) {
            throw std::runtime_error(
                "The metric returned a non-finite score"
            );
        }

        score_sum += score;
    }

    result.mean_score =
        score_sum
        / static_cast<double>(result.fold_scores.size());

    double squared_difference_sum = 0.0;

    for (double score : result.fold_scores) {
        const double difference =
            score - result.mean_score;

        squared_difference_sum +=
            difference * difference;
    }

    // Population standard deviation: the folds represent the entire
    // set of cross-validation scores being summarized.
    result.standard_deviation = std::sqrt(
        squared_difference_sum
        / static_cast<double>(result.fold_scores.size())
    );

    return result;
}

} // namespace chiikaml::model_selection