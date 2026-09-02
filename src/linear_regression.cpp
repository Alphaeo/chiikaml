#include "chiikaml/linear_regression.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

namespace chiikaml {

LinearRegression::LinearRegression(bool fit_intercept)
    : fit_intercept_(fit_intercept),
      fitted_(false),
      n_features_(0),
      intercept_(0.0) {}

void LinearRegression::fit(
    const Matrix& X,
    const std::vector<double>& y
) {
    const std::size_t n_samples = X.rows();
    const std::size_t n_features = X.cols();

    if (n_samples == 0 || n_features == 0 || y.empty()) {
        throw std::invalid_argument(
            "The training dataset must not be empty"
        );
    }

    if (n_samples != y.size()) {
        throw std::invalid_argument(
            "X.rows() must be equal to y.size()"
        );
    }

    const std::size_t minimum_samples =
    n_features + (fit_intercept_ ? 1 : 0);

    if (n_samples < minimum_samples) {
        throw std::invalid_argument(
            "There are not enough samples to fit the model"
        );
    }
    std::vector<double> feature_means(n_features, 0.0);
    double target_mean = 0.0;

    // Compute the feature and target means when an intercept is used.
    if (fit_intercept_) {
        for (std::size_t i = 0; i < n_samples; ++i) {
            target_mean += y[i];

            for (std::size_t j = 0; j < n_features; ++j) {
                feature_means[j] += X(i, j);
            }
        }

        const double inverse_sample_count =
            1.0 / static_cast<double>(n_samples);

        target_mean *= inverse_sample_count;

        for (double& mean : feature_means) {
            mean *= inverse_sample_count;
        }
    }

    // Construct the normal-equation matrix X^T X and right-hand
    // side X^T y directly, without explicitly creating X^T.
    Matrix gram(n_features, n_features);
    Matrix right_hand_side(n_features, 1);

    for (std::size_t i = 0; i < n_samples; ++i) {
        const double centered_y =
            y[i] - (fit_intercept_ ? target_mean : 0.0);

        for (std::size_t j = 0; j < n_features; ++j) {
            const double centered_x_j =
                X(i, j)
                - (fit_intercept_ ? feature_means[j] : 0.0);

            right_hand_side(j, 0) += centered_x_j * centered_y;

            // Compute only the lower triangle and mirror it because
            // X^T X is symmetric.
            for (std::size_t k = 0; k <= j; ++k) {
                const double centered_x_k =
                    X(i, k)
                    - (fit_intercept_ ? feature_means[k] : 0.0);

                gram(j, k) += centered_x_j * centered_x_k;
            }
        }
    }

    // Copy the lower triangle into the upper triangle.
    for (std::size_t i = 0; i < n_features; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            gram(j, i) = gram(i, j);
        }
    }

    // Solve (X^T X) * coefficients = X^T y.
    Matrix solution = gram.solve_cholesky(right_hand_side);

    std::vector<double> new_coefficients(n_features);

    for (std::size_t j = 0; j < n_features; ++j) {
        new_coefficients[j] = solution(j, 0);
    }

    double new_intercept = 0.0;

    if (fit_intercept_) {
        new_intercept = target_mean;

        for (std::size_t j = 0; j < n_features; ++j) {
            new_intercept -=
                feature_means[j] * new_coefficients[j];
        }
    }

    // Commit the fitted parameters only after training succeeds.
    coefficients_ = std::move(new_coefficients);
    intercept_ = new_intercept;
    n_features_ = n_features;
    fitted_ = true;
}

std::vector<double> LinearRegression::predict(
    const Matrix& X
) const {
    if (!fitted_) {
        throw std::runtime_error(
            "The model must be fitted before prediction"
        );
    }

    if (X.cols() != n_features_) {
        throw std::invalid_argument(
            "X has an incorrect number of features"
        );
    }

    std::vector<double> predictions(X.rows());

    for (std::size_t i = 0; i < X.rows(); ++i) {
        predictions[i] = predict_one(X, i);
    }

    return predictions;
}

const std::vector<double>&
LinearRegression::coefficients() const {
    if (!fitted_) {
        throw std::runtime_error(
            "The model has not been fitted"
        );
    }

    return coefficients_;
}

double LinearRegression::intercept() const {
    if (!fitted_) {
        throw std::runtime_error(
            "The model has not been fitted"
        );
    }

    return intercept_;
}

double LinearRegression::predict_one(
    const Matrix& X,
    std::size_t sample_row
) const {
    double prediction = intercept_;

    for (std::size_t j = 0; j < n_features_; ++j) {
        prediction += X(sample_row, j) * coefficients_[j];
    }

    return prediction;
}

} // namespace chiikaml