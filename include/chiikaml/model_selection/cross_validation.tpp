#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace chiikaml::model_selection {

namespace cross_validation_detail {

// Selects target values using the provided row indices.
inline std::vector<int> select_targets(
    const std::vector<int>& targets,
    const std::vector<std::size_t>& indices
) {
    std::vector<int> selected_targets;
    selected_targets.reserve(indices.size());

    for (std::size_t index : indices) {
        if (index >= targets.size()) {
            throw std::out_of_range(
                "Target index is outside the vector"
            );
        }

        selected_targets.push_back(targets[index]);
    }

    return selected_targets;
}

} // namespace cross_validation_detail

template <typename Model>
CrossValidationResult cross_validate_classification(
    const Model& model,
    const Matrix& X,
    const std::vector<int>& y,
    const KFold& splitter,
    const ClassificationMetric& metric
) {
    if (X.rows() == 0) {
        throw std::invalid_argument(
            "The dataset must not be empty"
        );
    }

    if (X.rows() != y.size()) {
        throw std::invalid_argument(
            "X.rows() must be equal to y.size()"
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
            cross_validation_detail::select_targets(
                y,
                training_indices
            );

        std::vector<int> y_validation =
            cross_validation_detail::select_targets(
                y,
                validation_indices
            );

        // Every fold receives an independent model copy. Fitted
        // state from one fold therefore cannot affect another fold.
        Model fold_model = model;

        fold_model.fit(
            X_train,
            y_train
        );

        const std::vector<int> predictions =
            fold_model.predict(X_validation);

        if (predictions.size() != y_validation.size()) {
            throw std::runtime_error(
                "The model returned an incorrect number of predictions"
            );
        }

        const double score = metric(
            y_validation,
            predictions
        );

        if (!std::isfinite(score)) {
            throw std::runtime_error(
                "The metric returned a non-finite score"
            );
        }

        result.fold_scores.push_back(score);
    }

    double score_sum = 0.0;

    for (double score : result.fold_scores) {
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

    result.standard_deviation = std::sqrt(
        squared_difference_sum
        / static_cast<double>(result.fold_scores.size())
    );

    return result;
}

} // namespace chiikaml::model_selection