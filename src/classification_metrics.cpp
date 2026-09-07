#include "chiikaml/metrics/classification_metrics.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace chiikaml::metrics {

namespace {

// Verifie les conditions communes a toutes les metriques.
void validate_targets(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred
) {
    if (y_true.empty()) {
        throw std::invalid_argument(
            "Target vectors must not be empty"
        );
    }

    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument(
            "y_true and y_pred must have the same size"
        );
    }
}

} // namespace

double accuracy(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred
) {
    validate_targets(y_true, y_pred);

    std::size_t correct_predictions = 0;

    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_pred[i]) {
            ++correct_predictions;
        }
    }

    return static_cast<double>(correct_predictions)
         / static_cast<double>(y_true.size());
}

double precision(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred
) {
    validate_targets(y_true, y_pred);

    constexpr int positive_label = 1;

    std::size_t true_positives = 0;
    std::size_t false_positives = 0;

    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (y_pred[i] == positive_label) {
            if (y_true[i] == positive_label) {
                ++true_positives;
            } else {
                ++false_positives;
            }
        }
    }

    const std::size_t predicted_positives =
        true_positives + false_positives;

    // Precision is defined as zero when the model never predicts
    // the positive class.
    if (predicted_positives == 0) {
        return 0.0;
    }

    return static_cast<double>(true_positives)
         / static_cast<double>(predicted_positives);
}

double recall(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred
) {
    validate_targets(y_true, y_pred);

    constexpr int positive_label = 1;

    std::size_t true_positives = 0;
    std::size_t false_negatives = 0;

    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == positive_label) {
            if (y_pred[i] == positive_label) {
                ++true_positives;
            } else {
                ++false_negatives;
            }
        }
    }

    const std::size_t actual_positives =
        true_positives + false_negatives;

    // Recall is defined as zero when the true data contains no
    // positive samples.
    if (actual_positives == 0) {
        return 0.0;
    }

    return static_cast<double>(true_positives)
         / static_cast<double>(actual_positives);
}

double f1_score(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred
) {
    validate_targets(y_true, y_pred);

    constexpr int positive_label = 1;

    std::size_t true_positives = 0;
    std::size_t false_positives = 0;
    std::size_t false_negatives = 0;

    // Compute all required counts in one traversal instead of
    // separately calling precision() and recall().
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == positive_label &&
            y_pred[i] == positive_label) {
            ++true_positives;
        } else if (y_true[i] != positive_label &&
                   y_pred[i] == positive_label) {
            ++false_positives;
        } else if (y_true[i] == positive_label &&
                   y_pred[i] != positive_label) {
            ++false_negatives;
        }
    }

    const std::size_t denominator =
        2 * true_positives
        + false_positives
        + false_negatives;

    if (denominator == 0) {
        return 0.0;
    }

    return 2.0 * static_cast<double>(true_positives)
         / static_cast<double>(denominator);
}

Matrix confusion_matrix(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred
) {
    validate_targets(y_true, y_pred);

    // Gather every label appearing in either vector.
    std::vector<int> labels;
    labels.reserve(y_true.size() + y_pred.size());

    labels.insert(
        labels.end(),
        y_true.begin(),
        y_true.end()
    );

    labels.insert(
        labels.end(),
        y_pred.begin(),
        y_pred.end()
    );

    std::sort(labels.begin(), labels.end());

    labels.erase(
        std::unique(labels.begin(), labels.end()),
        labels.end()
    );

    Matrix result(labels.size(), labels.size());

    for (std::size_t i = 0; i < y_true.size(); ++i) {
        const auto true_iterator = std::lower_bound(
            labels.begin(),
            labels.end(),
            y_true[i]
        );

        const auto predicted_iterator = std::lower_bound(
            labels.begin(),
            labels.end(),
            y_pred[i]
        );

        const std::size_t true_index =
            static_cast<std::size_t>(
                true_iterator - labels.begin()
            );

        const std::size_t predicted_index =
            static_cast<std::size_t>(
                predicted_iterator - labels.begin()
            );

        // Rows represent true labels and columns represent
        // predicted labels.
        result(true_index, predicted_index) += 1.0;
    }

    return result;
}

} // namespace chiikaml::metrics