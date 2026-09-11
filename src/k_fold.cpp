#include "chiikaml/model_selection/k_fold.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace chiikaml::model_selection {

KFold::KFold(
    std::size_t n_splits,
    bool shuffle,
    unsigned int seed
)
    : n_splits_(n_splits),
      shuffle_(shuffle),
      seed_(seed) {
    if (n_splits_ < 2) {
        throw std::invalid_argument(
            "n_splits must be at least 2"
        );
    }
}

std::vector<FoldIndices> KFold::split(
    std::size_t n_samples
) const {
    if (n_samples < n_splits_) {
        throw std::invalid_argument(
            "n_samples must be greater than or equal to n_splits"
        );
    }

    // Initially generate the indices:
    //
    //     0, 1, 2, ..., n_samples - 1
    std::vector<std::size_t> indices(n_samples);

    std::iota(
        indices.begin(),
        indices.end(),
        0
    );

    // Shuffle the indices before constructing the folds when
    // requested. Using a fixed seed makes the split reproducible.
    if (shuffle_) {
        std::mt19937 generator(seed_);

        std::shuffle(
            indices.begin(),
            indices.end(),
            generator
        );
    }

    std::vector<FoldIndices> folds;
    folds.reserve(n_splits_);

    const std::size_t base_fold_size =
        n_samples / n_splits_;

    const std::size_t remainder =
        n_samples % n_splits_;

    std::size_t validation_begin = 0;

    for (std::size_t fold = 0;
         fold < n_splits_;
         ++fold) {
        // If the samples cannot be distributed equally, the first
        // folds receive one additional validation sample.
        const std::size_t validation_size =
            base_fold_size + (fold < remainder ? 1 : 0);

        const std::size_t validation_end =
            validation_begin + validation_size;

        std::vector<std::size_t> training_indices;
        std::vector<std::size_t> validation_indices;

        training_indices.reserve(
            n_samples - validation_size
        );

        validation_indices.reserve(
            validation_size
        );

        // The current contiguous section of the potentially shuffled
        // index vector becomes the validation fold.
        validation_indices.insert(
            validation_indices.end(),
            indices.begin() + validation_begin,
            indices.begin() + validation_end
        );

        // Every index outside that section belongs to the training
        // set.
        training_indices.insert(
            training_indices.end(),
            indices.begin(),
            indices.begin() + validation_begin
        );

        training_indices.insert(
            training_indices.end(),
            indices.begin() + validation_end,
            indices.end()
        );

        folds.emplace_back(
            std::move(training_indices),
            std::move(validation_indices)
        );

        validation_begin = validation_end;
    }

    return folds;
}

std::size_t KFold::n_splits() const {
    return n_splits_;
}

bool KFold::shuffle() const {
    return shuffle_;
}

unsigned int KFold::seed() const {
    return seed_;
}

} // namespace chiikaml::model_selection