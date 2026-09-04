#include "chiikaml/binary_svm.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace chiikaml {

BinarySVM::BinarySVM(
    double C,
    SVMKernel kernel,
    double gamma,
    std::size_t degree,
    double coef0,
    std::size_t max_iterations,
    double tolerance,
    bool fit_intercept,
    unsigned int seed
)
    : C_(C),
      kernel_(kernel),
      configured_gamma_(gamma),
      gamma_(gamma),
      degree_(degree),
      coef0_(coef0),
      max_iterations_(max_iterations),
      tolerance_(tolerance),
      fit_intercept_(fit_intercept),
      seed_(seed),
      fitted_(false),
      converged_(false),
      iterations_(0),
      n_features_(0),
      support_vectors_(0, 0),
      intercept_(0.0) {
    if (!std::isfinite(C_) || C_ <= 0.0) {
        throw std::invalid_argument(
            "C must be a finite positive number"
        );
    }

    if (!std::isfinite(configured_gamma_) ||
        configured_gamma_ < 0.0) {
        throw std::invalid_argument(
            "Gamma must be a finite non-negative number"
        );
    }

    if (degree_ == 0) {
        throw std::invalid_argument(
            "Polynomial degree must be greater than zero"
        );
    }

    if (!std::isfinite(coef0_)) {
        throw std::invalid_argument(
            "Coef0 must be finite"
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

void BinarySVM::fit(
    const Matrix& X,
    const std::vector<int>& y
) {
    const std::size_t n_samples = X.rows();
    const std::size_t n_features = X.cols();

    if (n_samples == 0 || n_features == 0) {
        throw std::invalid_argument(
            "The training dataset must not be empty"
        );
    }

    if (n_samples != y.size()) {
        throw std::invalid_argument(
            "X.rows() must be equal to y.size()"
        );
    }

    bool has_negative_class = false;
    bool has_positive_class = false;

    for (std::size_t i = 0; i < n_samples; ++i) {
        if (y[i] == -1) {
            has_negative_class = true;
        } else if (y[i] == 1) {
            has_positive_class = true;
        } else {
            throw std::invalid_argument(
                "BinarySVM labels must be either -1 or +1"
            );
        }

        for (std::size_t j = 0; j < n_features; ++j) {
            if (!std::isfinite(X(i, j))) {
                throw std::invalid_argument(
                    "Training data must contain only finite values"
                );
            }
        }
    }

    if (!has_negative_class || !has_positive_class) {
        throw std::invalid_argument(
            "Both classes -1 and +1 must be present"
        );
    }

    const double effective_gamma =
        configured_gamma_ == 0.0
            ? 1.0 / static_cast<double>(n_features)
            : configured_gamma_;

    // Kernel function using the effective gamma selected for this fit.
    const auto kernel_value =
        [this, effective_gamma](
            const Matrix& first,
            std::size_t first_row,
            const Matrix& second,
            std::size_t second_row
        ) {
            if (kernel_ == SVMKernel::Linear) {
                double dot_product = 0.0;

                for (std::size_t j = 0;
                     j < first.cols();
                     ++j) {
                    dot_product +=
                        first(first_row, j)
                        * second(second_row, j);
                }

                return dot_product;
            }

            if (kernel_ == SVMKernel::Polynomial) {
                double dot_product = 0.0;

                for (std::size_t j = 0;
                     j < first.cols();
                     ++j) {
                    dot_product +=
                        first(first_row, j)
                        * second(second_row, j);
                }

                return std::pow(
                    effective_gamma * dot_product + coef0_,
                    static_cast<double>(degree_)
                );
            }

            double squared_distance = 0.0;

            for (std::size_t j = 0;
                 j < first.cols();
                 ++j) {
                const double difference =
                    first(first_row, j)
                    - second(second_row, j);

                squared_distance += difference * difference;
            }

            return std::exp(
                -effective_gamma * squared_distance
            );
        };

    // Precompute the complete symmetric kernel matrix.
    //
    // This requires O(n_samples^2) memory but avoids repeatedly
    // evaluating expensive RBF or polynomial kernels during SMO.
    Matrix kernel_matrix(n_samples, n_samples);

    for (std::size_t i = 0; i < n_samples; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            const double value =
                kernel_value(X, i, X, j);

            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "The kernel produced a non-finite value"
                );
            }

            kernel_matrix(i, j) = value;
            kernel_matrix(j, i) = value;
        }
    }

    std::vector<double> alphas(n_samples, 0.0);
    double new_intercept = 0.0;

    std::mt19937 generator(seed_);

    std::vector<std::size_t> sample_order(n_samples);
    std::iota(
        sample_order.begin(),
        sample_order.end(),
        0
    );

    bool new_converged = false;
    std::size_t performed_iterations = 0;

    if (fit_intercept_) {
        // Initially f(x_i) = 0, therefore:
        //
        //     error_i = f(x_i) - y_i = -y_i
        std::vector<double> errors(n_samples);

        for (std::size_t i = 0; i < n_samples; ++i) {
            errors[i] = -static_cast<double>(y[i]);
        }

        // Simplified Platt SMO.
        for (std::size_t iteration = 0;
             iteration < max_iterations_;
             ++iteration) {
            std::shuffle(
                sample_order.begin(),
                sample_order.end(),
                generator
            );

            std::size_t changed_pairs = 0;

            for (std::size_t position = 0;
                 position < n_samples;
                 ++position) {
                const std::size_t i = sample_order[position];

                const double y_i =
                    static_cast<double>(y[i]);

                const double error_i = errors[i];

                const bool violates_lower_bound =
                    y_i * error_i < -tolerance_
                    && alphas[i] < C_;

                const bool violates_upper_bound =
                    y_i * error_i > tolerance_
                    && alphas[i] > 0.0;

                if (!violates_lower_bound &&
                    !violates_upper_bound) {
                    continue;
                }

                // Select the second sample using the maximum error
                // difference heuristic.
                std::size_t j = n_samples;
                double largest_error_difference = -1.0;

                for (std::size_t candidate = 0;
                     candidate < n_samples;
                     ++candidate) {
                    if (candidate == i) {
                        continue;
                    }

                    const double difference = std::abs(
                        error_i - errors[candidate]
                    );

                    if (difference >
                        largest_error_difference) {
                        largest_error_difference = difference;
                        j = candidate;
                    }
                }

                if (j == n_samples) {
                    continue;
                }

                const double y_j =
                    static_cast<double>(y[j]);

                const double error_j = errors[j];

                const double old_alpha_i = alphas[i];
                const double old_alpha_j = alphas[j];

                double lower_bound = 0.0;
                double upper_bound = 0.0;

                if (y[i] != y[j]) {
                    lower_bound = std::max(
                        0.0,
                        old_alpha_j - old_alpha_i
                    );

                    upper_bound = std::min(
                        C_,
                        C_ + old_alpha_j - old_alpha_i
                    );
                } else {
                    lower_bound = std::max(
                        0.0,
                        old_alpha_i + old_alpha_j - C_
                    );

                    upper_bound = std::min(
                        C_,
                        old_alpha_i + old_alpha_j
                    );
                }

                if (lower_bound >= upper_bound) {
                    continue;
                }

                const double eta =
                    2.0 * kernel_matrix(i, j)
                    - kernel_matrix(i, i)
                    - kernel_matrix(j, j);

                // For a positive semi-definite kernel, eta should
                // normally be strictly negative.
                if (eta >= 0.0) {
                    continue;
                }

                double new_alpha_j =
                    old_alpha_j
                    - y_j * (error_i - error_j) / eta;

                new_alpha_j = std::clamp(
                    new_alpha_j,
                    lower_bound,
                    upper_bound
                );

                const double alpha_change_threshold =
                    std::numeric_limits<double>::epsilon()
                    * std::max(1.0, C_);

                if (std::abs(
                        new_alpha_j - old_alpha_j
                    ) <= alpha_change_threshold) {
                    continue;
                }

                const double new_alpha_i =
                    old_alpha_i
                    + y_i * y_j
                    * (old_alpha_j - new_alpha_j);

                const double delta_i =
                    new_alpha_i - old_alpha_i;

                const double delta_j =
                    new_alpha_j - old_alpha_j;

                const double old_intercept = new_intercept;

                const double first_intercept =
                    old_intercept
                    - error_i
                    - y_i * delta_i
                        * kernel_matrix(i, i)
                    - y_j * delta_j
                        * kernel_matrix(i, j);

                const double second_intercept =
                    old_intercept
                    - error_j
                    - y_i * delta_i
                        * kernel_matrix(i, j)
                    - y_j * delta_j
                        * kernel_matrix(j, j);

                if (new_alpha_i > 0.0 &&
                    new_alpha_i < C_) {
                    new_intercept = first_intercept;
                } else if (new_alpha_j > 0.0 &&
                           new_alpha_j < C_) {
                    new_intercept = second_intercept;
                } else {
                    new_intercept =
                        0.5
                        * (first_intercept + second_intercept);
                }

                alphas[i] = new_alpha_i;
                alphas[j] = new_alpha_j;

                const double intercept_change =
                    new_intercept - old_intercept;

                // Update every cached training error incrementally.
                for (std::size_t sample = 0;
                     sample < n_samples;
                     ++sample) {
                    errors[sample] +=
                        y_i * delta_i
                            * kernel_matrix(i, sample)
                        + y_j * delta_j
                            * kernel_matrix(j, sample)
                        + intercept_change;
                }

                ++changed_pairs;
            }

            performed_iterations = iteration + 1;

            if (changed_pairs == 0) {
                new_converged = true;
                break;
            }
        }
    } else {
        // Without an intercept, the dual problem does not contain
        // the equality constraint sum(alpha_i * y_i) = 0.
        //
        // It can therefore be optimized one coordinate at a time.
        std::vector<double> scores(n_samples, 0.0);

        for (std::size_t iteration = 0;
             iteration < max_iterations_;
             ++iteration) {
            std::shuffle(
                sample_order.begin(),
                sample_order.end(),
                generator
            );

            double maximum_change = 0.0;

            for (std::size_t position = 0;
                 position < n_samples;
                 ++position) {
                const std::size_t i = sample_order[position];

                const double y_i =
                    static_cast<double>(y[i]);

                const double old_alpha = alphas[i];

                // Dual derivative with respect to alpha_i.
                const double gradient =
                    1.0 - y_i * scores[i];

                double new_alpha = old_alpha;
                const double diagonal = kernel_matrix(i, i);

                if (diagonal > 0.0) {
                    new_alpha = std::clamp(
                        old_alpha + gradient / diagonal,
                        0.0,
                        C_
                    );
                } else if (gradient > 0.0) {
                    new_alpha = C_;
                } else {
                    new_alpha = 0.0;
                }

                const double change =
                    new_alpha - old_alpha;

                if (change == 0.0) {
                    continue;
                }

                alphas[i] = new_alpha;

                for (std::size_t sample = 0;
                     sample < n_samples;
                     ++sample) {
                    scores[sample] +=
                        change * y_i
                        * kernel_matrix(i, sample);
                }

                maximum_change = std::max(
                    maximum_change,
                    std::abs(change)
                );
            }

            performed_iterations = iteration + 1;

            if (maximum_change <=
                tolerance_ * std::max(1.0, C_)) {
                new_converged = true;
                break;
            }
        }

        new_intercept = 0.0;
    }

    const double support_threshold =
        100.0
        * std::numeric_limits<double>::epsilon()
        * std::max(1.0, C_);

    std::size_t support_count = 0;

    for (double alpha : alphas) {
        if (alpha > support_threshold) {
            ++support_count;
        }
    }

    if (support_count == 0) {
        throw std::runtime_error(
            "SVM training produced no support vectors"
        );
    }

    Matrix new_support_vectors(
        support_count,
        n_features
    );

    std::vector<double> new_dual_coefficients;
    new_dual_coefficients.reserve(support_count);

    std::size_t destination_row = 0;

    for (std::size_t i = 0; i < n_samples; ++i) {
        if (alphas[i] <= support_threshold) {
            continue;
        }

        for (std::size_t j = 0; j < n_features; ++j) {
            new_support_vectors(destination_row, j) =
                X(i, j);
        }

        new_dual_coefficients.push_back(
            alphas[i] * static_cast<double>(y[i])
        );

        ++destination_row;
    }

    support_vectors_ = std::move(new_support_vectors);
    dual_coefficients_ =
        std::move(new_dual_coefficients);

    intercept_ = new_intercept;
    gamma_ = effective_gamma;
    n_features_ = n_features;
    iterations_ = performed_iterations;
    converged_ = new_converged;
    fitted_ = true;
}

std::vector<int> BinarySVM::predict(
    const Matrix& X
) const {
    validate_prediction_data(X);

    std::vector<int> predictions(X.rows());

    for (std::size_t i = 0; i < X.rows(); ++i) {
        predictions[i] =
            decision_value(X, i) >= 0.0 ? 1 : -1;
    }

    return predictions;
}

std::vector<double> BinarySVM::decision_function(
    const Matrix& X
) const {
    validate_prediction_data(X);

    std::vector<double> scores(X.rows());

    for (std::size_t i = 0; i < X.rows(); ++i) {
        scores[i] = decision_value(X, i);
    }

    return scores;
}

double BinarySVM::decision_value(
    const Matrix& X,
    std::size_t sample_row
) const {
    validate_prediction_data(X);

    if (sample_row >= X.rows()) {
        throw std::out_of_range(
            "Sample row is outside the matrix"
        );
    }

    double score = intercept_;

    for (std::size_t i = 0;
         i < support_vectors_.rows();
         ++i) {
        score +=
            dual_coefficients_[i]
            * evaluate_kernel(
                support_vectors_,
                i,
                X,
                sample_row
            );
    }

    return score;
}

const Matrix& BinarySVM::support_vectors() const {
    if (!fitted_) {
        throw std::runtime_error(
            "The model has not been fitted"
        );
    }

    return support_vectors_;
}

const std::vector<double>&
BinarySVM::dual_coefficients() const {
    if (!fitted_) {
        throw std::runtime_error(
            "The model has not been fitted"
        );
    }

    return dual_coefficients_;
}

double BinarySVM::intercept() const {
    if (!fitted_) {
        throw std::runtime_error(
            "The model has not been fitted"
        );
    }

    return intercept_;
}

double BinarySVM::C() const {
    return C_;
}

SVMKernel BinarySVM::kernel() const {
    return kernel_;
}

double BinarySVM::gamma() const {
    if (!fitted_) {
        if (configured_gamma_ == 0.0) {
            throw std::runtime_error(
                "Automatic gamma is only known after fit()"
            );
        }

        return configured_gamma_;
    }

    return gamma_;
}

std::size_t BinarySVM::degree() const {
    return degree_;
}

double BinarySVM::coef0() const {
    return coef0_;
}

bool BinarySVM::converged() const {
    if (!fitted_) {
        throw std::runtime_error(
            "The model has not been fitted"
        );
    }

    return converged_;
}

std::size_t BinarySVM::iterations() const {
    if (!fitted_) {
        throw std::runtime_error(
            "The model has not been fitted"
        );
    }

    return iterations_;
}

std::size_t
BinarySVM::number_of_support_vectors() const {
    if (!fitted_) {
        throw std::runtime_error(
            "The model has not been fitted"
        );
    }

    return support_vectors_.rows();
}

double BinarySVM::evaluate_kernel(
    const Matrix& first,
    std::size_t first_row,
    const Matrix& second,
    std::size_t second_row
) const {
    if (kernel_ == SVMKernel::Linear) {
        double dot_product = 0.0;

        for (std::size_t j = 0;
             j < n_features_;
             ++j) {
            dot_product +=
                first(first_row, j)
                * second(second_row, j);
        }

        return dot_product;
    }

    if (kernel_ == SVMKernel::Polynomial) {
        double dot_product = 0.0;

        for (std::size_t j = 0;
             j < n_features_;
             ++j) {
            dot_product +=
                first(first_row, j)
                * second(second_row, j);
        }

        return std::pow(
            gamma_ * dot_product + coef0_,
            static_cast<double>(degree_)
        );
    }

    double squared_distance = 0.0;

    for (std::size_t j = 0;
         j < n_features_;
         ++j) {
        const double difference =
            first(first_row, j)
            - second(second_row, j);

        squared_distance += difference * difference;
    }

    return std::exp(-gamma_ * squared_distance);
}

void BinarySVM::validate_prediction_data(
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
}

} // namespace chiikaml::detail