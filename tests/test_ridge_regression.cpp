#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

#include "chiikaml/matrix.hpp"
#include "chiikaml/ridge_regression.hpp"
#include "chiikaml/linear_regression.hpp"

using chiikaml::Matrix;
using chiikaml::RidgeRegression;
using chiikaml::LinearRegression;

TEST_CASE("RidgeRegression learns the correct regularized coefficients and intercept",
          "[ridge_regression]") {
    Matrix X(3, 1);

    X(0, 0) = 1;
    X(1, 0) = 2;
    X(2, 0) = 3;

    // Original relationship:
    //
    //     y = 2*x + 1
    //
    // With alpha = 2:
    //
    //     coefficient = 1
    //     intercept = 3
    std::vector<double> y = {
        3,
        5,
        7
    };

    RidgeRegression model(2.0);
    model.fit(X, y);

    REQUIRE(model.alpha() == 2.0);
    REQUIRE(model.coefficients().size() == 1);

    REQUIRE(
        model.coefficients()[0]
        == Catch::Approx(1.0).margin(1e-12)
    );

    REQUIRE(
        model.intercept()
        == Catch::Approx(3.0).margin(1e-12)
    );
}

TEST_CASE("RidgeRegression predicts new values",
          "[ridge_regression]") {
    Matrix X_train(3, 1);

    X_train(0, 0) = 1;
    X_train(1, 0) = 2;
    X_train(2, 0) = 3;

    std::vector<double> y = {
        3,
        5,
        7
    };

    RidgeRegression model(2.0);
    model.fit(X_train, y);

    Matrix X_test(3, 1);

    X_test(0, 0) = 1;
    X_test(1, 0) = 2;
    X_test(2, 0) = 4;

    std::vector<double> predictions =
        model.predict(X_test);

    REQUIRE(predictions.size() == 3);

    REQUIRE(
        predictions[0]
        == Catch::Approx(4.0).margin(1e-12)
    );

    REQUIRE(
        predictions[1]
        == Catch::Approx(5.0).margin(1e-12)
    );

    REQUIRE(
        predictions[2]
        == Catch::Approx(7.0).margin(1e-12)
    );
}

TEST_CASE("RidgeRegression can be fitted without an intercept",
          "[ridge_regression]") {
    Matrix X(2, 1);

    X(0, 0) = 1;
    X(1, 0) = 2;

    std::vector<double> y = {
        2,
        4
    };

    // X^T X = 5
    // X^T y = 10
    // alpha = 5
    //
    // coefficient = 10 / (5 + 5) = 1
    RidgeRegression model(5.0, false);
    model.fit(X, y);

    REQUIRE(model.coefficients().size() == 1);

    REQUIRE(
        model.coefficients()[0]
        == Catch::Approx(1.0).margin(1e-12)
    );

    REQUIRE(
        model.intercept()
        == Catch::Approx(0.0).margin(1e-12)
    );
}

TEST_CASE("RidgeRegression handles linearly dependent features",
          "[ridge_regression]") {
    Matrix X(3, 2);

    // The two features are identical, so X^T X is singular.
    X(0, 0) = 0; X(0, 1) = 0;
    X(1, 0) = 1; X(1, 1) = 1;
    X(2, 0) = 2; X(2, 1) = 2;

    std::vector<double> y = {
        0,
        2,
        4
    };

    // The regularization makes X^T X + alpha*I
    // positive definite despite the dependent features.
    RidgeRegression model(1.0);

    REQUIRE_NOTHROW(model.fit(X, y));

    REQUIRE(model.coefficients().size() == 2);

    REQUIRE(
        model.coefficients()[0]
        == Catch::Approx(0.8).margin(1e-12)
    );

    REQUIRE(
        model.coefficients()[1]
        == Catch::Approx(0.8).margin(1e-12)
    );

    REQUIRE(
        model.intercept()
        == Catch::Approx(0.4).margin(1e-12)
    );
}

TEST_CASE("RidgeRegression with alpha zero behaves like LinearRegression",
          "[ridge_regression]") {
    Matrix X_train(5, 2);

    X_train(0, 0) = 0; X_train(0, 1) = 0;
    X_train(1, 0) = 1; X_train(1, 1) = 0;
    X_train(2, 0) = 0; X_train(2, 1) = 1;
    X_train(3, 0) = 2; X_train(3, 1) = 1;
    X_train(4, 0) = 1; X_train(4, 1) = 3;

    // Exact relationship:
    //
    //     y = 2*x1 - 3*x2 + 5
    std::vector<double> y_train = {
        5,
        7,
        2,
        6,
        -2
    };

    LinearRegression linear_model;
    RidgeRegression ridge_model(0.0);

    linear_model.fit(X_train, y_train);
    ridge_model.fit(X_train, y_train);

    const std::vector<double>& linear_coefficients =
        linear_model.coefficients();

    const std::vector<double>& ridge_coefficients =
        ridge_model.coefficients();

    REQUIRE(
        ridge_coefficients.size()
        == linear_coefficients.size()
    );

    for (std::size_t j = 0;
         j < linear_coefficients.size();
         ++j) {
        REQUIRE(
            ridge_coefficients[j]
            == Catch::Approx(linear_coefficients[j])
                   .margin(1e-12)
        );
    }

    REQUIRE(
        ridge_model.intercept()
        == Catch::Approx(linear_model.intercept())
               .margin(1e-12)
    );
}

TEST_CASE("RidgeRegression rejects incompatible training dimensions",
          "[ridge_regression]") {
    Matrix X(3, 1);
    std::vector<double> y = {1, 2};

    RidgeRegression model;

    REQUIRE_THROWS_AS(
        model.fit(X, y),
        std::invalid_argument
    );
}

TEST_CASE("RidgeRegression requires fitting before prediction",
          "[ridge_regression]") {
    RidgeRegression model;
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

TEST_CASE("RidgeRegression rejects an incorrect number of features",
          "[ridge_regression]") {
    Matrix X_train(3, 1);

    X_train(0, 0) = 1;
    X_train(1, 0) = 2;
    X_train(2, 0) = 3;

    std::vector<double> y = {
        3,
        5,
        7
    };

    RidgeRegression model;
    model.fit(X_train, y);

    Matrix X_test(2, 2);

    REQUIRE_THROWS_AS(
        model.predict(X_test),
        std::invalid_argument
    );
}