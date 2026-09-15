#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace chiikaml::model_selection {

// Indices generated for one fold:
//
//     first  = training indices
//     second = validation indices
using FoldIndices = std::pair<
    std::vector<std::size_t>,
    std::vector<std::size_t>
>;

// K-fold cross-validation splitter.
//
// The dataset is divided into n_splits folds. For each split, one
// fold is used as the validation set and all remaining folds form
// the training set.
//
// This class only generates row indices. It does not copy matrices,
// train models or compute metrics.
class KFold {
public:
    // n_splits must be at least 2.
    //
    // If shuffle is true, sample indices are shuffled before folds
    // are generated. The seed makes the result reproducible.
    explicit KFold(
        std::size_t n_splits = 5,
        bool shuffle = false,
        unsigned int seed = 42
    );

    // Generates the training and validation indices for every fold.
    //
    // n_samples must be greater than or equal to n_splits.
    //
    // When n_samples is not divisible by n_splits, the first folds
    // receive one additional validation sample.
    std::vector<FoldIndices> split(
        std::size_t n_samples
    ) const;

    std::size_t n_splits() const;
    bool shuffle() const;
    unsigned int seed() const;

private:
    std::size_t n_splits_;
    bool shuffle_;
    unsigned int seed_;
};

} // namespace chiikaml::model_selection