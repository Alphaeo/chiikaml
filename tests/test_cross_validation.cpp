#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "chiikaml/knn.hpp"
#include "chiikaml/matrix.hpp"
#include "chiikaml/metrics/classification_metrics.hpp"
#include "chiikaml/model_selection/cross_validation.hpp"
#include "chiikaml/model_selection/k_fold.hpp"

using Catch::Approx;
using chiikaml::KNNClassifier;
using chiikaml::Matrix;
using chiikaml::model_selection::KFold;
using chiikaml::model_selection::
    cross_validate_classification;

namespace {

// Creates two clearly separated classes.
//
// Class 0 is located around x = 0.
// Class 1 is located around x = 10.
Matrix create_dataset() {
    Matrix X(10, 1);

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

    return X;
}

std::vector<int> create_targets() {
    return {
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1
    };
}

// Deliberately returns an incorrect number of predictions.
class InvalidClassifier {
public:
    void fit(
        const Matrix&,
        const std::vector<int>&
    ) {}

    std::vector<int> predict(
        const Matrix&
    ) const {
        return {};
    }
};

} // namespace

TEST_CASE(
    "Cross-validation returns one score for every fold",
    "[cross_validation]"
) {
    const Matrix X = create_dataset();
    const std::vector<int> y = create_targets();

    const KNNClassifier model(1);
    const KFold splitter(5, false, 42);

    const auto result = cross_validate_classification(
        model,
        X,
        y,
        splitter,
        chiikaml::metrics::accuracy
    );

    REQUIRE(result.fold_scores.size() == 5);
}

TEST_CASE(
    "Cross-validation correctly evaluates a separable dataset",
    "[cross_validation]"
) {
    const Matrix X = create_dataset();
    const std::vector<int> y = create_targets();

    const KNNClassifier model(1);
    const KFold splitter(5, false, 42);

    const auto result = cross_validate_classification(
        model,
        X,
        y,
        splitter,
        chiikaml::metrics::accuracy
    );

    for (double score : result.fold_scores) {
        REQUIRE(score == Approx(1.0));
    }

    REQUIRE(result.mean_score == Approx(1.0));
    REQUIRE(result.standard_deviation == Approx(0.0));
}

TEST_CASE(
    "Cross-validation works with shuffled folds",
    "[cross_validation]"
) {
    const Matrix X = create_dataset();
    const std::vector<int> y = create_targets();

    const KNNClassifier model(1);
    const KFold splitter(5, true, 42);

    const auto first_result =
        cross_validate_classification(
            model,
            X,
            y,
            splitter,
            chiikaml::metrics::accuracy
        );

    const auto second_result =
        cross_validate_classification(
            model,
            X,
            y,
            splitter,
            chiikaml::metrics::accuracy
        );

    REQUIRE(
        first_result.fold_scores.size()
        == second_result.fold_scores.size()
    );

    for (std::size_t fold = 0;
         fold < first_result.fold_scores.size();
         ++fold) {
        REQUIRE(
            first_result.fold_scores[fold]
            == Approx(second_result.fold_scores[fold])
        );
    }

    REQUIRE(
        first_result.mean_score
        == Approx(second_result.mean_score)
    );

    REQUIRE(
        first_result.standard_deviation
        == Approx(second_result.standard_deviation)
    );
}

TEST_CASE(
    "Cross-validation supports different classification metrics",
    "[cross_validation]"
) {
    const Matrix X = create_dataset();
    const std::vector<int> y = create_targets();

    const KNNClassifier model(1);
    const KFold splitter(5, true, 42);

    const auto result = cross_validate_classification(
        model,
        X,
        y,
        splitter,
        chiikaml::metrics::f1_score
    );

    REQUIRE(result.fold_scores.size() == 5);

    for (double score : result.fold_scores) {
        REQUIRE(score >= 0.0);
        REQUIRE(score <= 1.0);
    }

    REQUIRE(result.mean_score >= 0.0);
    REQUIRE(result.mean_score <= 1.0);
}

TEST_CASE(
    "Cross-validation rejects mismatched X and y sizes",
    "[cross_validation]"
) {
    const Matrix X = create_dataset();

    const std::vector<int> invalid_targets = {
        0, 0, 0
    };

    const KNNClassifier model(1);
    const KFold splitter(5, false, 42);

    REQUIRE_THROWS_AS(
        cross_validate_classification(
            model,
            X,
            invalid_targets,
            splitter,
            chiikaml::metrics::accuracy
        ),
        std::invalid_argument
    );
}

TEST_CASE(
    "Cross-validation rejects an empty dataset",
    "[cross_validation]"
) {
    const Matrix X(0, 1);
    const std::vector<int> y;

    const KNNClassifier model(1);
    const KFold splitter(2, false, 42);

    REQUIRE_THROWS_AS(
        cross_validate_classification(
            model,
            X,
            y,
            splitter,
            chiikaml::metrics::accuracy
        ),
        std::invalid_argument
    );
}

TEST_CASE(
    "Cross-validation detects an invalid prediction count",
    "[cross_validation]"
) {
    const Matrix X = create_dataset();
    const std::vector<int> y = create_targets();

    const InvalidClassifier model;
    const KFold splitter(5, false, 42);

    REQUIRE_THROWS_AS(
        cross_validate_classification(
            model,
            X,
            y,
            splitter,
            chiikaml::metrics::accuracy
        ),
        std::runtime_error
    );
}