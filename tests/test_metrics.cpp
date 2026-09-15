#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

#include "chiikaml/metrics/classification_metrics.hpp"
#include "chiikaml/metrics/regression_metrics.hpp"



using chiikaml::Matrix;
using namespace chiikaml::metrics;

TEST_CASE("Classification metrics are correctly computed",
          "[metrics][classification]") {
    // Confusion counts:
    //
    // True positives:  2
    // False positives: 1
    // False negatives: 2
    // True negatives:  2
    const std::vector<int> y_true = {
        1, 1, 1, 1, 0, 0, 0
    };

    const std::vector<int> y_pred = {
        1, 1, 0, 0, 1, 0, 0
    };

    REQUIRE(
        accuracy(y_true, y_pred)
        == Catch::Approx(4.0 / 7.0)
    );

    REQUIRE(
        precision(y_true, y_pred)
        == Catch::Approx(2.0 / 3.0)
    );

    REQUIRE(
        recall(y_true, y_pred)
        == Catch::Approx(1.0 / 2.0)
    );

    REQUIRE(
        f1_score(y_true, y_pred)
        == Catch::Approx(4.0 / 7.0)
    );
}





TEST_CASE("Binary confusion matrix is correctly computed",
          "[metrics][classification]") {
    const std::vector<int> y_true = {
        1, 1, 1, 1, 0, 0, 0
    };

    const std::vector<int> y_pred = {
        1, 1, 0, 0, 1, 0, 0
    };

    const Matrix result =
        confusion_matrix(y_true, y_pred);

    REQUIRE(result.rows() == 2);
    REQUIRE(result.cols() == 2);

    // Labels are sorted as [0, 1].
    //
    // Rows:    true labels
    // Columns: predicted labels
    //
    //              predicted
    //              0    1
    // true 0       2    1
    // true 1       2    2
    REQUIRE(result(0, 0) == 2);
    REQUIRE(result(0, 1) == 1);
    REQUIRE(result(1, 0) == 2);
    REQUIRE(result(1, 1) == 2);
}

TEST_CASE("Confusion matrix supports arbitrary multiclass labels",
          "[metrics][classification]") {
    const std::vector<int> y_true = {
        5, 2, 9, 5
    };

    const std::vector<int> y_pred = {
        2, 2, 5, 9
    };

    const Matrix result =
        confusion_matrix(y_true, y_pred);

    REQUIRE(result.rows() == 3);
    REQUIRE(result.cols() == 3);

    // Labels are sorted as [2, 5, 9].
    //
    //              predicted
    //              2    5    9
    // true 2       1    0    0
    // true 5       1    0    1
    // true 9       0    1    0
    REQUIRE(result(0, 0) == 1);
    REQUIRE(result(0, 1) == 0);
    REQUIRE(result(0, 2) == 0);

    REQUIRE(result(1, 0) == 1);
    REQUIRE(result(1, 1) == 0);
    REQUIRE(result(1, 2) == 1);

    REQUIRE(result(2, 0) == 0);
    REQUIRE(result(2, 1) == 1);
    REQUIRE(result(2, 2) == 0);
}

TEST_CASE("Classification metrics reject empty targets",
          "[metrics][classification]") {
    const std::vector<int> y_true;
    const std::vector<int> y_pred;

    REQUIRE_THROWS_AS(
        accuracy(y_true, y_pred),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        precision(y_true, y_pred),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        recall(y_true, y_pred),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        f1_score(y_true, y_pred),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        confusion_matrix(y_true, y_pred),
        std::invalid_argument
    );
}

TEST_CASE("Classification metrics reject different target sizes",
          "[metrics][classification]") {
    const std::vector<int> y_true = {
        0, 1, 1
    };

    const std::vector<int> y_pred = {
        0, 1
    };

    REQUIRE_THROWS_AS(
        accuracy(y_true, y_pred),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        precision(y_true, y_pred),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        recall(y_true, y_pred),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        f1_score(y_true, y_pred),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        confusion_matrix(y_true, y_pred),
        std::invalid_argument
    );
}

// Regression metrics tests

TEST_CASE(
    "Regression metrics correctly evaluate basic predictions",
    "[regression_metrics]"
) {
    const std::vector<double> y_true = {
        1.0, 2.0, 3.0
    };

    const std::vector<double> y_pred = {
        1.0, 3.0, 2.0
    };

    REQUIRE(
        chiikaml::metrics::mean_squared_error(
            y_true,
            y_pred
        ) == Catch::Approx(2.0 / 3.0)
    );

    REQUIRE(
        chiikaml::metrics::mean_absolute_error(
            y_true,
            y_pred
        ) == Catch::Approx(2.0 / 3.0)
    );

    REQUIRE(
        chiikaml::metrics::r2_score(
            y_true,
            y_pred
        ) == Catch::Approx(0.0)
    );
}

TEST_CASE(
    "Regression metrics recognize perfect predictions",
    "[regression_metrics]"
) {
    const std::vector<double> y_true = {
        2.0, 4.0, 6.0, 8.0
    };

    const std::vector<double> y_pred = y_true;

    REQUIRE(
        chiikaml::metrics::mean_squared_error(
            y_true,
            y_pred
        ) == Catch::Approx(0.0)
    );

    REQUIRE(
        chiikaml::metrics::mean_absolute_error(
            y_true,
            y_pred
        ) == Catch::Approx(0.0)
    );

    REQUIRE(
        chiikaml::metrics::r2_score(
            y_true,
            y_pred
        ) == Catch::Approx(1.0)
    );
}

TEST_CASE(
    "Regression metrics reject invalid target vectors",
    "[regression_metrics]"
) {
    const std::vector<double> empty;

    REQUIRE_THROWS_AS(
        chiikaml::metrics::mean_squared_error(
            empty,
            empty
        ),
        std::invalid_argument
    );

    const std::vector<double> y_true = {
        1.0, 2.0, 3.0
    };

    const std::vector<double> y_pred = {
        1.0, 2.0
    };

    REQUIRE_THROWS_AS(
        chiikaml::metrics::mean_absolute_error(
            y_true,
            y_pred
        ),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        chiikaml::metrics::r2_score(
            y_true,
            y_pred
        ),
        std::invalid_argument
    );
}