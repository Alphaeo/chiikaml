#include "chiikaml/model_selection/train_test_split.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace chiikaml::model_selection {

template<typename Target>
TrainTestSplitResult<Target> train_test_split(
    const Matrix& X,
    const std::vector<Target>& y,
    double train_size,
    bool shuffle,
    unsigned int seed
) {
    double test_size = 1.0 - train_size;
    if (X.rows() != y.size()) {
        throw std::invalid_argument(
            "X.rows() must be equal to y.size()"
        );
    }

    if (X.rows() < 2) {
        throw std::invalid_argument(
            "The dataset must contain at least two samples"
        );
    }

    if (!std::isfinite(test_size) ||
        test_size <= 0.0 ||
        test_size >= 1.0) {
        throw std::invalid_argument(
            "test_size must be strictly between 0 and 1"
        );
    }

    const std::size_t n_samples = X.rows();
    const std::size_t n_features = X.cols();

    // Convert the requested proportion into a number of test
    // samples. At least one sample is always assigned to the test set.
    const std::size_t n_test = std::max<std::size_t>(
        1,
        static_cast<std::size_t>(
            test_size * static_cast<double>(n_samples)
        )
    );

    const std::size_t n_train = n_samples - n_test;

    // Create the row indices of the original dataset.
    std::vector<std::size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle only the indices. This avoids moving complete matrix
    // rows during the shuffle operation.
    if (shuffle) {
        std::mt19937 generator(seed);

        std::shuffle(
            indices.begin(),
            indices.end(),
            generator
        );
    }

    Matrix X_train(n_train, n_features);
    Matrix X_test(n_test, n_features);

    std::vector<Target> y_train(n_train);
    std::vector<Target> y_test(n_test);

    // The first n_train indices are assigned to the training set.
    for (std::size_t destination_row = 0;
         destination_row < n_train;
         ++destination_row) {
        const std::size_t source_row =
            indices[destination_row];

        for (std::size_t column = 0;
             column < n_features;
             ++column) {
            X_train(destination_row, column) =
                X(source_row, column);
        }

        y_train[destination_row] = y[source_row];
    }

    // The remaining indices are assigned to the test set.
    for (std::size_t destination_row = 0;
         destination_row < n_test;
         ++destination_row) {
        const std::size_t source_row =
            indices[n_train + destination_row];

        for (std::size_t column = 0;
             column < n_features;
             ++column) {
            X_test(destination_row, column) =
                X(source_row, column);
        }

        y_test[destination_row] = y[source_row];
    }

    return {
        std::move(X_train),
        std::move(X_test),
        std::move(y_train),
        std::move(y_test)
    };
}

// Generate the classification version of train_test_split.
template TrainTestSplitResult<int> train_test_split<int>(
    const Matrix& X,
    const std::vector<int>& y,
    double test_size,
    bool shuffle,
    unsigned int seed
);

// Generate the regression version of train_test_split.
template TrainTestSplitResult<double> train_test_split<double>(
    const Matrix& X,
    const std::vector<double>& y,
    double test_size,
    bool shuffle,
    unsigned int seed
);

} // namespace chiikaml::model_selection