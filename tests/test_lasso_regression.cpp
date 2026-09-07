#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

#include "chiikaml/lasso_regression.hpp"
#include "chiikaml/linear_regression.hpp"
#include "chiikaml/matrix.hpp"

using chiikaml::LassoRegression;
using chiikaml::LinearRegression;
using chiikaml::Matrix;

TEST_CASE("LassoRegression shrinks coefficients and selects features",
          "[lasso_regression]") {
    Matrix X(4, 2);

    // The centered feature columns are orthogonal.
    X(0, 0) = -1; X(0, 1) = -1;
    X(1, 0) = -1; X(1, 1) =  1;
    X(2, 0) =  1; X(2, 1) = -1;
    X(3, 0) =  1; X(3, 1) =  1;

    // y = 3*x1 + 0.2*x2 + 5
    //
    // With alpha = 1:
    // - the first coefficient is reduced from 3 to 2.75;
    // - the second coefficient becomes exactly zero.
    std::vector<double> y = {
        1.8,
        2.2,
        7.8,
        8.2
    };

    LassoRegression model(1.0);
    model.fit(X, y);

    REQUIRE(model.converged());
    REQUIRE(model.iterations() <= 1000);
    REQUIRE(model.coefficients().size() == 2);

    REQUIRE(
        model.coefficients()[0]
        == Catch::Approx(2.75).margin(1e-12)
    );

    REQUIRE(model.coefficients()[1] == 0.0);

    REQUIRE(
        model.intercept()
        == Catch::Approx(5.0).margin(1e-12)
    );
}

TEST_CASE("LassoRegression predicts new values",
          "[lasso_regression]") {
    Matrix X_train(4, 2);

    X_train(0, 0) = -1; X_train(0, 1) = -1;
    X_train(1, 0) = -1; X_train(1, 1) =  1;
    X_train(2, 0) =  1; X_train(2, 1) = -1;
    X_train(3, 0) =  1; X_train(3, 1) =  1;

    std::vector<double> y = {
        1.8,
        2.2,
        7.8,
        8.2
    };

    LassoRegression model(1.0);
    model.fit(X_train, y);

    Matrix X_test(3, 2);

    X_test(0, 0) = -1; X_test(0, 1) = 10;
    X_test(1, 0) =  0; X_test(1, 1) = -5;
    X_test(2, 0) =  1; X_test(2, 1) = 20;

    const std::vector<double> predictions =
        model.predict(X_test);

    REQUIRE(predictions.size() == 3);

    // The second feature has coefficient zero.
    REQUIRE(
        predictions[0]
        == Catch::Approx(2.25).margin(1e-12)
    );

    REQUIRE(
        predictions[1]
        == Catch::Approx(5.0).margin(1e-12)
    );

    REQUIRE(
        predictions[2]
        == Catch::Approx(7.75).margin(1e-12)
    );
}

TEST_CASE("LassoRegression works without an intercept",
          "[lasso_regression]") {
    Matrix X(2, 1);

    X(0, 0) = 1;
    X(1, 0) = 2;

    std::vector<double> y = {
        2,
        4
    };

    // X^T X = 5 and X^T y = 10.
    //
    // coefficient = soft_threshold(10, 5) / 5 = 1.
    LassoRegression model(5.0, false);
    model.fit(X, y);

    REQUIRE(model.converged());

    REQUIRE(
        model.coefficients()[0]
        == Catch::Approx(1.0).margin(1e-12)
    );

    REQUIRE(
        model.intercept()
        == Catch::Approx(0.0).margin(1e-12)
    );
}

TEST_CASE("LassoRegression with alpha zero behaves like LinearRegression",
          "[lasso_regression]") {
    Matrix X_train(4, 2);

    // Orthogonal features let coordinate descent reach the exact
    // ordinary least-squares solution immediately.
    X_train(0, 0) = -1; X_train(0, 1) = -1;
    X_train(1, 0) = -1; X_train(1, 1) =  1;
    X_train(2, 0) =  1; X_train(2, 1) = -1;
    X_train(3, 0) =  1; X_train(3, 1) =  1;

    std::vector<double> y_train = {
        1.8,
        2.2,
        7.8,
        8.2
    };

    LinearRegression linear_model;
    LassoRegression lasso_model(0.0);

    linear_model.fit(X_train, y_train);
    lasso_model.fit(X_train, y_train);

    REQUIRE(lasso_model.converged());

    REQUIRE(
        lasso_model.coefficients().size()
        == linear_model.coefficients().size()
    );

    for (std::size_t j = 0;
         j < linear_model.coefficients().size();
         ++j) {
        REQUIRE(
            lasso_model.coefficients()[j]
            == Catch::Approx(
                linear_model.coefficients()[j]
            ).margin(1e-12)
        );
    }

    REQUIRE(
        lasso_model.intercept()
        == Catch::Approx(
            linear_model.intercept()
        ).margin(1e-12)
    );

    Matrix X_test(3, 2);

    X_test(0, 0) = 2;  X_test(0, 1) = 3;
    X_test(1, 0) = -2; X_test(1, 1) = 4;
    X_test(2, 0) = 0;  X_test(2, 1) = -3;

    const std::vector<double> linear_predictions =
        linear_model.predict(X_test);

    const std::vector<double> lasso_predictions =
        lasso_model.predict(X_test);

    REQUIRE(
        lasso_predictions.size()
        == linear_predictions.size()
    );

    for (std::size_t i = 0;
         i < linear_predictions.size();
         ++i) {
        REQUIRE(
            lasso_predictions[i]
            == Catch::Approx(
                linear_predictions[i]
            ).margin(1e-12)
        );
    }
}



TEST_CASE("LassoRegression rejects invalid parameters",
          "[lasso_regression]") {
    REQUIRE_THROWS_AS(
        LassoRegression(-1.0),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        LassoRegression(1.0, true, 0),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        LassoRegression(1.0, true, 1000, 0.0),
        std::invalid_argument
    );
}

TEST_CASE("LassoRegression rejects incompatible training dimensions",
          "[lasso_regression]") {
    Matrix X(3, 1);
    std::vector<double> y = {1, 2};

    LassoRegression model;

    REQUIRE_THROWS_AS(
        model.fit(X, y),
        std::invalid_argument
    );
}

TEST_CASE("LassoRegression requires fitting before prediction",
          "[lasso_regression]") {
    LassoRegression model;
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

    REQUIRE_THROWS_AS(
        model.converged(),
        std::runtime_error
    );

    REQUIRE_THROWS_AS(
        model.iterations(),
        std::runtime_error
    );
}

TEST_CASE("LassoRegression rejects an incorrect number of features",
          "[lasso_regression]") {
    Matrix X_train(3, 1);

    X_train(0, 0) = 1;
    X_train(1, 0) = 2;
    X_train(2, 0) = 3;

    std::vector<double> y = {3, 5, 7};

    LassoRegression model;
    model.fit(X_train, y);

    Matrix X_test(2, 2);

    REQUIRE_THROWS_AS(
        model.predict(X_test),
        std::invalid_argument
    );
}