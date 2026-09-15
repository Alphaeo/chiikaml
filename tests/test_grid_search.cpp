#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <vector>

#include "chiikaml/knn.hpp"
#include "chiikaml/matrix.hpp"
#include "chiikaml/metrics/classification_metrics.hpp"
#include "chiikaml/model_selection/grid_search.hpp"
#include "chiikaml/model_selection/k_fold.hpp"

namespace {

struct KNNParameters {
    std::size_t k;
};

}

TEST_CASE(
    "Grid Search evaluates KNN configurations",
    "[grid_search]"
) {
    chiikaml::Matrix X(10, 1);

    X(0, 0) = 0.0;
    X(1, 0) = 0.1;
    X(2, 0) = 0.2;
    X(3, 0) = 0.3;
    X(4, 0) = 0.4;

    X(5, 0) = 10.0;
    X(6, 0) = 10.1;
    X(7, 0) = 10.2;
    X(8, 0) = 10.3;
    X(9, 0) = 10.4;

    const std::vector<int> y = {
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1
    };

    const std::vector<KNNParameters> parameter_grid = {
        {1},
        {3}
    };

    const auto result =
        chiikaml::model_selection::
            grid_search_classification(
                parameter_grid,
                X,
                y,
                chiikaml::model_selection::KFold(
                    5,
                    false,
                    42
                ),
                [](const KNNParameters& parameters) {
                    return chiikaml::KNNClassifier(
                        parameters.k
                    );
                },
                chiikaml::metrics::accuracy
            );

    REQUIRE(result.candidate_results.size() == 2);
    REQUIRE(result.best_score == 1.0);

    // Both values may obtain the same score. Grid Search keeps the
    // first configuration when scores are tied.
    REQUIRE(result.best_index == 0);
    REQUIRE(result.best_parameters.k == 1);
}

TEST_CASE(
    "Grid Search returns a fitted best KNN model",
    "[grid_search]"
) {
    chiikaml::Matrix X(6, 1);

    X(0, 0) = 0.0;
    X(1, 0) = 0.1;
    X(2, 0) = 0.2;

    X(3, 0) = 10.0;
    X(4, 0) = 10.1;
    X(5, 0) = 10.2;

    const std::vector<int> y = {
        0, 0, 0,
        1, 1, 1
    };

    const std::vector<KNNParameters> parameter_grid = {
        {1}
    };

    const auto result =
        chiikaml::model_selection::
            grid_search_classification(
                parameter_grid,
                X,
                y,
                chiikaml::model_selection::KFold(
                    3,
                    true,
                    42
                ),
                [](const KNNParameters& parameters) {
                    return chiikaml::KNNClassifier(
                        parameters.k
                    );
                },
                chiikaml::metrics::accuracy
            );

    const std::vector<int> predictions =
        result.best_model.predict(X);

    REQUIRE(predictions == y);
}