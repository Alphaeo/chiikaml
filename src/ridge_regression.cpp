#include "chiikaml/ridge_regression.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace chiikaml {

RidgeRegression::RidgeRegression(
    double alpha,
    bool fit_intercept
)
    : alpha_(alpha),
      fit_intercept_(fit_intercept),
      fitted_(false),
      n_features_(0),
      intercept_(0.0) {
    if (!std::isfinite(alpha_) || alpha_ < 0.0) {
        throw std::invalid_argument(
            "Alpha must be a finite non-negative number"
        );
    }
}

void RidgeRegression::fit(
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

    // When alpha is zero, Ridge becomes ordinary linear regression.
    // In this case, enough samples are needed for a unique solution.
    const std::size_t minimum_samples =
        n_features + (fit_intercept_ ? 1 : 0);

    if (alpha_ == 0.0 && n_samples < minimum_samples) {
        throw std::invalid_argument(
            "There are not enough samples to fit the model "
            "without regularization"
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

    // Construct the regularized normal-equation matrix:
    //
    //     X^T X + alpha * I
    //
    // and the right-hand side:
    //
    //     X^T y
    //
    // Only the lower triangle of the Gram matrix is computed because
    // the matrix is symmetric.
    Matrix gram(n_features, n_features);
    Matrix right_hand_side(n_features, 1);

    // Reused for every sample to avoid recomputing centered values
    // several times inside the Gram-matrix calculation.
    std::vector<double> centered_row(n_features);

    for (std::size_t i = 0; i < n_samples; ++i) {
        const double centered_y =
            y[i] - (fit_intercept_ ? target_mean : 0.0);

        for (std::size_t j = 0; j < n_features; ++j) {
            centered_row[j] =
                X(i, j)
                - (fit_intercept_ ? feature_means[j] : 0.0);
        }

        for (std::size_t j = 0; j < n_features; ++j) {
            right_hand_side(j, 0) +=
                centered_row[j] * centered_y;

            for (std::size_t k = 0; k <= j; ++k) {
                gram(j, k) +=
                    centered_row[j] * centered_row[k];
            }
        }
    }

    // Add the L2 regularization to the diagonal.
    // The intercept is not present in this matrix because the data
    // was centered, so it is not regularized.
    for (std::size_t j = 0; j < n_features; ++j) {
        gram(j, j) += alpha_;
    }

    // Copy the lower triangle into the upper triangle.
    for (std::size_t i = 0; i < n_features; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            gram(j, i) = gram(i, j);
        }
    }

    // Solve:
    //
    //     (X^T X + alpha * I) * coefficients = X^T y
    Matrix solution =
        gram.solve_cholesky(right_hand_side);

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

    // Update the model only after training has completed successfully.
    coefficients_ = std::move(new_coefficients);
    intercept_ = new_intercept;
    n_features_ = n_features;
    fitted_ = true;
}

std::vector<double> RidgeRegression::predict(
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
RidgeRegression::coefficients() const {
    if (!fitted_) {
        throw std::runtime_error(
            "The model has not been fitted"
        );
    }

    return coefficients_;
}

double RidgeRegression::intercept() const {
    if (!fitted_) {
        throw std::runtime_error(
            "The model has not been fitted"
        );
    }

    return intercept_;
}

double RidgeRegression::alpha() const {
    return alpha_;
}

double RidgeRegression::predict_one(
    const Matrix& X,
    std::size_t sample_row
) const {
    double prediction = intercept_;

    for (std::size_t j = 0; j < n_features_; ++j) {
        prediction +=
            X(sample_row, j) * coefficients_[j];
    }

    return prediction;
}

} // namespace chiikaml