#include "chiikaml/metrics/regression_metrics.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace chiikaml::metrics {

namespace {

void validate_targets(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred
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

double mean_squared_error(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred
) {
    validate_targets(y_true, y_pred);

    double sum_squared_errors = 0.0;

    for (std::size_t i = 0; i < y_true.size(); ++i) {
        const double error = y_true[i] - y_pred[i];
        sum_squared_errors += error * error;
    }

    return sum_squared_errors
         / static_cast<double>(y_true.size());
}

double mean_absolute_error(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred
) {
    validate_targets(y_true, y_pred);

    double sum_absolute_errors = 0.0;

    for (std::size_t i = 0; i < y_true.size(); ++i) {
        sum_absolute_errors += std::abs(
            y_true[i] - y_pred[i]
        );
    }

    return sum_absolute_errors
         / static_cast<double>(y_true.size());
}

double r2_score(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred
) {
    validate_targets(y_true, y_pred);

    const double mean_y_true =
        std::accumulate(
            y_true.begin(),
            y_true.end(),
            0.0
        ) / static_cast<double>(y_true.size());

    double total_sum_of_squares = 0.0;
    double residual_sum_of_squares = 0.0;

    for (std::size_t i = 0; i < y_true.size(); ++i) {
        const double total_error =
            y_true[i] - mean_y_true;

        const double residual_error =
            y_true[i] - y_pred[i];

        total_sum_of_squares +=
            total_error * total_error;

        residual_sum_of_squares +=
            residual_error * residual_error;
    }

    // R² is mathematically undefined for a constant target. Use the
    // practical finite convention also used by many ML workflows:
    //
    //     perfect predictions     -> 1
    //     imperfect predictions   -> 0
    if (total_sum_of_squares == 0.0) {
        return residual_sum_of_squares == 0.0
            ? 1.0
            : 0.0;
    }

    return 1.0
         - residual_sum_of_squares
         / total_sum_of_squares;
}

} // namespace chiikaml::metrics