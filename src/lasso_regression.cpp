#include "chiikaml/lasso_regression.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace chiikaml {

namespace {

// Proximal operator of the L1 penalty.
//
// Values inside [-threshold, threshold] become exactly zero.
// Other values are moved toward zero by threshold.
double soft_threshold(double value, double threshold) {
    if (value > threshold) {
        return value - threshold;
    }

    if (value < -threshold) {
        return value + threshold;
    }

    return 0.0;
}

} // namespace

LassoRegression::LassoRegression(
    double alpha,
    bool fit_intercept,
    std::size_t max_iterations,
    double tolerance
)
    : alpha_(alpha),
      fit_intercept_(fit_intercept),
      max_iterations_(max_iterations),
      tolerance_(tolerance),
      fitted_(false),
      converged_(false),
      iterations_(0),
      n_features_(0),
      intercept_(0.0) {
    if (!std::isfinite(alpha_) || alpha_ < 0.0) {
        throw std::invalid_argument(
            "Alpha must be a finite non-negative number"
        );
    }

    if (max_iterations_ == 0) {
        throw std::invalid_argument(
            "Max iterations must be greater than zero"
        );
    }

    if (!std::isfinite(tolerance_) || tolerance_ <= 0.0) {
        throw std::invalid_argument(
            "Tolerance must be a finite positive number"
        );
    }
}

void LassoRegression::fit(
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

    std::vector<double> feature_means(n_features, 0.0);
    double target_mean = 0.0;

    // Compute feature and target means when fitting an intercept.
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

    // Coordinate descent repeatedly traverses feature columns.
    // Store centered X in column-major order so that every column
    // is contiguous in memory during the hot loops.
    std::vector<double> centered_columns(
        n_samples * n_features
    );

    // Squared norm of every centered feature column.
    std::vector<double> squared_norms(n_features, 0.0);

    for (std::size_t j = 0; j < n_features; ++j) {
        const std::size_t column_offset = j * n_samples;

        for (std::size_t i = 0; i < n_samples; ++i) {
            const double centered_value =
                X(i, j)
                - (fit_intercept_ ? feature_means[j] : 0.0);

            centered_columns[column_offset + i] =
                centered_value;

            squared_norms[j] +=
                centered_value * centered_value;
        }
    }

    // Start with zero coefficients. The initial residual is
    // therefore equal to the centered target.
    std::vector<double> new_coefficients(n_features, 0.0);
    std::vector<double> residual(n_samples);

    for (std::size_t i = 0; i < n_samples; ++i) {
        residual[i] =
            y[i] - (fit_intercept_ ? target_mean : 0.0);
    }

    bool new_converged = false;
    std::size_t performed_iterations = 0;

    // Cyclic coordinate descent.
    for (std::size_t iteration = 0;
         iteration < max_iterations_;
         ++iteration) {
        double maximum_change = 0.0;

        for (std::size_t j = 0; j < n_features; ++j) {
            const std::size_t column_offset =
                j * n_samples;

            const double old_coefficient =
                new_coefficients[j];

            // Add the old contribution of feature j back into the
            // residual before recomputing its coefficient.
            if (old_coefficient != 0.0) {
                for (std::size_t i = 0;
                     i < n_samples;
                     ++i) {
                    residual[i] +=
                        centered_columns[column_offset + i]
                        * old_coefficient;
                }
            }

            double correlation = 0.0;

            for (std::size_t i = 0;
                 i < n_samples;
                 ++i) {
                correlation +=
                    centered_columns[column_offset + i]
                    * residual[i];
            }

            double new_coefficient = 0.0;

            // A zero-norm column is constant after centering and
            // therefore carries no usable information.
            if (squared_norms[j] > 0.0) {
                new_coefficient =
                    soft_threshold(correlation, alpha_)
                    / squared_norms[j];
            }

            new_coefficients[j] = new_coefficient;

            // Remove the contribution of the new coefficient from
            // the residual.
            if (new_coefficient != 0.0) {
                for (std::size_t i = 0;
                     i < n_samples;
                     ++i) {
                    residual[i] -=
                        centered_columns[column_offset + i]
                        * new_coefficient;
                }
            }

            maximum_change = std::max(
                maximum_change,
                std::abs(new_coefficient - old_coefficient)
            );
        }

        performed_iterations = iteration + 1;

        double coefficient_scale = 1.0;

        for (double coefficient : new_coefficients) {
            coefficient_scale = std::max(
                coefficient_scale,
                std::abs(coefficient)
            );
        }

        if (maximum_change <=
            tolerance_ * coefficient_scale) {
            new_converged = true;
            break;
        }
    }

    double new_intercept = 0.0;

    if (fit_intercept_) {
        new_intercept = target_mean;

        for (std::size_t j = 0; j < n_features; ++j) {
            new_intercept -=
                feature_means[j] * new_coefficients[j];
        }
    }

    // Commit the new model state after training has completed.
    coefficients_ = std::move(new_coefficients);
    intercept_ = new_intercept;
    n_features_ = n_features;
    iterations_ = performed_iterations;
    converged_ = new_converged;
    fitted_ = true;
}

std::vector<double> LassoRegression::predict(
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
LassoRegression::coefficients() const {
    if (!fitted_) {
        throw std::runtime_error(
            "The model has not been fitted"
        );
    }

    return coefficients_;
}

double LassoRegression::intercept() const {
    if (!fitted_) {
        throw std::runtime_error(
            "The model has not been fitted"
        );
    }

    return intercept_;
}

double LassoRegression::alpha() const {
    return alpha_;
}

bool LassoRegression::converged() const {
    if (!fitted_) {
        throw std::runtime_error(
            "The model has not been fitted"
        );
    }

    return converged_;
}

std::size_t LassoRegression::iterations() const {
    if (!fitted_) {
        throw std::runtime_error(
            "The model has not been fitted"
        );
    }

    return iterations_;
}

double LassoRegression::predict_one(
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