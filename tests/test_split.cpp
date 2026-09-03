#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <stdexcept>
#include <vector>

#include "chiikaml/matrix.hpp"
#include "chiikaml/model_selection/train_test_split.hpp"

using chiikaml::Matrix;
using chiikaml::model_selection::train_test_split;

TEST_CASE("train_test_split separates classification data with the correct split sizes",
          "[model_selection]") {
    Matrix X(5, 2);

    X(0, 0) = 10; X(0, 1) = 11;
    X(1, 0) = 20; X(1, 1) = 21;
    X(2, 0) = 30; X(2, 1) = 31;
    X(3, 0) = 40; X(3, 1) = 41;
    X(4, 0) = 50; X(4, 1) = 51;

    std::vector<int> y = {
        0, 1, 0, 1, 2
    };

    const auto split = train_test_split(
        X,
        y,
        0.8,
        false
    );

    REQUIRE(split.X_train.rows() == 4);
    REQUIRE(split.X_train.cols() == 2);
    REQUIRE(split.y_train.size() == 4);

    REQUIRE(split.X_test.rows() == 1);
    REQUIRE(split.X_test.cols() == 2);
    REQUIRE(split.y_test.size() == 1);

    // Without shuffling, the first four samples form the training set.
    REQUIRE(split.X_train(0, 0) == 10);
    REQUIRE(split.X_train(0, 1) == 11);
    REQUIRE(split.y_train[0] == 0);

    REQUIRE(split.X_train(3, 0) == 40);
    REQUIRE(split.X_train(3, 1) == 41);
    REQUIRE(split.y_train[3] == 1);

    // The last sample forms the test set.
    REQUIRE(split.X_test(0, 0) == 50);
    REQUIRE(split.X_test(0, 1) == 51);
    REQUIRE(split.y_test[0] == 2);
}

TEST_CASE("train_test_split supports regression targets",
          "[model_selection]") {
    Matrix X(4, 1);

    X(0, 0) = 1;
    X(1, 0) = 2;
    X(2, 0) = 3;
    X(3, 0) = 4;

    std::vector<double> y = {
        1.5,
        2.5,
        3.5,
        4.5
    };

    const auto split = train_test_split(
        X,
        y,
        0.5,
        false
    );

    REQUIRE(split.X_train.rows() == 2);
    REQUIRE(split.X_test.rows() == 2);

    REQUIRE(split.y_train.size() == 2);
    REQUIRE(split.y_test.size() == 2);

    REQUIRE(split.X_train(0, 0) == 1);
    REQUIRE(split.y_train[0] == 1.5);

    REQUIRE(split.X_train(1, 0) == 2);
    REQUIRE(split.y_train[1] == 2.5);

    REQUIRE(split.X_test(0, 0) == 3);
    REQUIRE(split.y_test[0] == 3.5);

    REQUIRE(split.X_test(1, 0) == 4);
    REQUIRE(split.y_test[1] == 4.5);
}



TEST_CASE("train_test_split is reproducible with the same seed",
          "[model_selection]") {
    Matrix X(10, 1);
    std::vector<int> y(10);

    for (std::size_t i = 0; i < 10; ++i) {
        X(i, 0) = static_cast<double>(i);
        y[i] = static_cast<int>(i);
    }

    const auto first = train_test_split(
        X,
        y,
        0.7,
        true,
        123
    );

    const auto second = train_test_split(
        X,
        y,
        0.7,
        true,
        123
    );

    REQUIRE(first.y_train == second.y_train);
    REQUIRE(first.y_test == second.y_test);

    for (std::size_t i = 0;
         i < first.X_train.rows();
         ++i) {
        REQUIRE(
            first.X_train(i, 0)
            == second.X_train(i, 0)
        );
    }

    for (std::size_t i = 0;
         i < first.X_test.rows();
         ++i) {
        REQUIRE(
            first.X_test(i, 0)
            == second.X_test(i, 0)
        );
    }
}

TEST_CASE("train_test_split rejects incompatible dimensions",
          "[model_selection]") {
    Matrix X(3, 1);

    const std::vector<int> y = {
        0, 1
    };

    REQUIRE_THROWS_AS(
        train_test_split(X, y),
        std::invalid_argument
    );
}

TEST_CASE("train_test_split requires at least two samples",
          "[model_selection]") {
    Matrix X(1, 1);
    const std::vector<int> y = {1};

    REQUIRE_THROWS_AS(
        train_test_split(X, y),
        std::invalid_argument
    );
}

TEST_CASE("train_test_split rejects invalid test sizes",
          "[model_selection]") {
    Matrix X(3, 1);
    const std::vector<int> y = {0, 1, 2};

    REQUIRE_THROWS_AS(
        train_test_split(X, y, 0.0),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        train_test_split(X, y, 1.0),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        train_test_split(X, y, -0.5),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        train_test_split(X, y, 1.5),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        train_test_split(
            X,
            y,
            std::numeric_limits<double>::quiet_NaN()
        ),
        std::invalid_argument
    );
}