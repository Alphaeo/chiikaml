#pragma once

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace chiikaml::model_selection {

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
> {
    using Model = std::decay_t<
        std::invoke_result_t<
            ModelFactory,
            const Parameters&
        >
    >;

    if (parameter_grid.empty()) {
        throw std::invalid_argument(
            "The parameter grid must not be empty"
        );
    }

    if (X.rows() == 0) {
        throw std::invalid_argument(
            "The dataset must not be empty"
        );
    }

    if (X.rows() != y.size()) {
        throw std::invalid_argument(
            "X.rows() must be equal to y.size()"
        );
    }

    if (!metric) {
        throw std::invalid_argument(
            "The metric function must be valid"
        );
    }

    std::vector<GridSearchCandidateResult<Parameters>>
        candidate_results;

    candidate_results.reserve(parameter_grid.size());

    // Evaluate the first configuration separately. This allows the
    // best parameters and score to be initialized without requiring
    // Parameters or Model to have a default constructor.
    Model first_model = std::invoke(
        make_model,
        parameter_grid[0]
    );

    CrossValidationResult first_result =
        cross_validate_classification(
            first_model,
            X,
            y,
            splitter,
            metric
        );

    candidate_results.push_back(
        GridSearchCandidateResult<Parameters>{
            parameter_grid[0],
            std::move(first_result)
        }
    );

    std::size_t best_index = 0;

    double best_score =
        candidate_results[0]
            .cross_validation
            .mean_score;

    // Evaluate every remaining parameter configuration.
    for (std::size_t candidate_index = 1;
         candidate_index < parameter_grid.size();
         ++candidate_index) {
        Model candidate_model = std::invoke(
            make_model,
            parameter_grid[candidate_index]
        );

        CrossValidationResult validation_result =
            cross_validate_classification(
                candidate_model,
                X,
                y,
                splitter,
                metric
            );

        const double candidate_score =
            validation_result.mean_score;

        candidate_results.push_back(
            GridSearchCandidateResult<Parameters>{
                parameter_grid[candidate_index],
                std::move(validation_result)
            }
        );

        // The first configuration wins in case of an exact tie.
        if (candidate_score > best_score) {
            best_score = candidate_score;
            best_index = candidate_index;
        }
    }

    // Construct a fresh model using the best parameters and refit it
    // on the complete dataset. This is the model returned to the
    // caller for future predictions.
    Model best_model = std::invoke(
        make_model,
        parameter_grid[best_index]
    );

    best_model.fit(X, y);

    return GridSearchResult<Model, Parameters>{
        std::move(best_model),
        parameter_grid[best_index],
        best_index,
        best_score,
        std::move(candidate_results)
    };
}

} // namespace chiikaml::model_selection