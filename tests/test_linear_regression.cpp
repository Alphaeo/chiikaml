#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

#include "chiikaml/linear_regression.hpp"
#include "chiikaml/matrix.hpp"

using chiikaml::LinearRegression;
using chiikaml::Matrix;

TEST_CASE("LinearRegression learns coefficients and intercept",
          "[linear_regression]") {
    Matrix X(5, 2);

    X(0, 0) = 0; X(0, 1) = 0;
    X(1, 0) = 1; X(1, 1) = 0;
    X(2, 0) = 0; X(2, 1) = 1;
    X(3, 0) = 2; X(3, 1) = 1;
    X(4, 0) = 1; X(4, 1) = 3;

    // Exact relationship:
    //
    //     y = 2*x1 - 3*x2 + 5
    std::vector<double> y = {
        5,
        7,
        2,
        6,
        -2
    };

    LinearRegression model;
    model.fit(X, y);

    const std::vector<double>& coefficients =
        model.coefficients();

    REQUIRE(coefficients.size() == 2);
    REQUIRE(coefficients[0] ==
            Catch::Approx(2.0).margin(1e-12));
    REQUIRE(coefficients[1] ==
            Catch::Approx(-3.0).margin(1e-12));
    REQUIRE(model.intercept() ==
            Catch::Approx(5.0).margin(1e-12));
}

TEST_CASE("LinearRegression predicts new values",
          "[linear_regression]") {
    Matrix X_train(4, 1);

    X_train(0, 0) = 1;
    X_train(1, 0) = 2;
    X_train(2, 0) = 3;
    X_train(3, 0) = 4;

    // y = 2*x + 1
    std::vector<double> y_train = {
        3,
        5,
        7,
        9
    };

    LinearRegression model;
    model.fit(X_train, y_train);

    Matrix X_test(3, 1);

    X_test(0, 0) = 5;
    X_test(1, 0) = 6;
    X_test(2, 0) = -1;

    std::vector<double> predictions =
        model.predict(X_test);

    REQUIRE(predictions.size() == 3);
    REQUIRE(predictions[0] ==
            Catch::Approx(11.0).margin(1e-12));
    REQUIRE(predictions[1] ==
            Catch::Approx(13.0).margin(1e-12));
    REQUIRE(predictions[2] ==
            Catch::Approx(-1.0).margin(1e-12));
}

TEST_CASE("LinearRegression can be fitted without an intercept",
          "[linear_regression]") {
    Matrix X(4, 1);

    X(0, 0) = 1;
    X(1, 0) = 2;
    X(2, 0) = 3;
    X(3, 0) = 4;

    // Exact relationship passing through the origin:
    //
    //     y = 3*x
    std::vector<double> y = {
        3,
        6,
        9,
        12
    };

    LinearRegression model(false);
    model.fit(X, y);

    REQUIRE(model.coefficients().size() == 1);
    REQUIRE(model.coefficients()[0] ==
            Catch::Approx(3.0).margin(1e-12));
    REQUIRE(model.intercept() ==
            Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("LinearRegression rejects incompatible training dimensions",
          "[linear_regression]") {
    Matrix X(3, 2);
    std::vector<double> y = {1, 2};

    LinearRegression model;

    REQUIRE_THROWS_AS(
        model.fit(X, y),
        std::invalid_argument
    );
}

TEST_CASE("LinearRegression requires fitting before prediction",
          "[linear_regression]") {
    LinearRegression model;
    Matrix X(2, 1);

    REQUIRE_THROWS_AS(
        model.predict(X),
        std::runtime_error
    );

    REQUIRE_THROWS_AS(
        model.coefficients(),
        std::runtime_error
    );

    REQUIRE_THROWS_AS(
        model.intercept(),
        std::runtime_error
    );
}

TEST_CASE("LinearRegression rejects an incorrect number of features",
          "[linear_regression]") {
    Matrix X_train(3, 1);

    X_train(0, 0) = 1;
    X_train(1, 0) = 2;
    X_train(2, 0) = 3;

    std::vector<double> y = {3, 5, 7};

    LinearRegression model;
    model.fit(X_train, y);

    Matrix X_test(2, 2);

    REQUIRE_THROWS_AS(
        model.predict(X_test),
        std::invalid_argument
    );
}

TEST_CASE("LinearRegression rejects linearly dependent features",
          "[linear_regression]") {
    Matrix X(4, 2);

    // The second feature is exactly twice the first one.
    X(0, 0) = 1; X(0, 1) = 2;
    X(1, 0) = 2; X(1, 1) = 4;
    X(2, 0) = 3; X(2, 1) = 6;
    X(3, 0) = 4; X(3, 1) = 8;

    std::vector<double> y = {1, 2, 3, 4};

    LinearRegression model;

    REQUIRE_THROWS_AS(
        model.fit(X, y),
        std::runtime_error
    );
}