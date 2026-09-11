#pragma once

#include <cstddef>
#include <type_traits>
#include <vector>

#include "chiikaml/matrix.hpp"
#include "chiikaml/model_selection/cross_validation.hpp"
#include "chiikaml/model_selection/k_fold.hpp"

namespace chiikaml::model_selection {

// Cross-validation result associated with one parameter
// configuration.
template <typename Parameters>
struct GridSearchCandidateResult {
    Parameters parameters;
    CrossValidationResult cross_validation;
};

// Complete result of Grid Search.
//
// best_model is retrained on the complete dataset after the best
// parameter configuration has been selected.
template <typename Model, typename Parameters>
struct GridSearchResult {
    Model best_model;
    Parameters best_parameters;
    std::size_t best_index;
    double best_score;

    std::vector<GridSearchCandidateResult<Parameters>>
        candidate_results;
};

// Evaluates every parameter configuration using cross-validation.
//
// make_model receives one parameter configuration and returns a new,
// unfitted model.
//
// The configuration with the highest mean cross-validation score is
// selected. Its model is then fitted on the complete dataset.
template <
    typename Parameters,
    typename ModelFactory
>
auto grid_search_classification(
    const std::vector<Parameters>& parameter_grid,
    const Matrix& X,
    const std::vector<int>& y,
    const KFold& splitter,
    ModelFactory make_model,
    const ClassificationMetric& metric
) -> GridSearchResult<
    std::decay_t<
        std::invoke_result_t<
            ModelFactory,
            const Parameters&
        >
    >,
    Parameters
>;

} // namespace chiikaml::model_selection

#include "chiikaml/model_selection/grid_search.tpp"