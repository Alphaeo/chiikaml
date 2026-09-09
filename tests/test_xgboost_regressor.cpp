#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <vector>

#include "chiikaml/matrix.hpp"
#include "chiikaml/xgboost_regressor.hpp"

using chiikaml::Matrix;
using chiikaml::XGBoostRegressor;

namespace {
double mse(const std::vector<double>& predictions, const std::vector<double>& y) {
    double total = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        double diff = predictions[i] - y[i];
        total += diff * diff;
    }
    return total / static_cast<double>(y.size());
}
} // namespace

TEST_CASE("Avec zero round, XGBoostRegressor predit simplement la moyenne de y", "[xgboost_regressor]") {
    Matrix X(4, 1);
    X(0, 0) = 1;
    X(1, 0) = 2;
    X(2, 0) = 3;
    X(3, 0) = 4;
    std::vector<double> y = {10.0, 20.0, 30.0, 40.0}; // moyenne = 25.0

    XGBoostRegressor model(/*n_estimators=*/0, /*learning_rate=*/0.3, /*max_depth=*/3);
    model.fit(X, y);

    auto preds = model.predict(X);
    for (double p : preds) {
        REQUIRE(p == Catch::Approx(25.0));
    }
}

TEST_CASE("XGBoostRegressor approche une relation lineaire simple", "[xgboost_regressor]") {
    Matrix X(6, 1);
    std::vector<double> y(6);
    for (std::size_t i = 0; i < 6; ++i) {
        X(i, 0) = static_cast<double>(i);
        y[i] = 2.0 * static_cast<double>(i) + 1.0; // y = 2x + 1
    }

    XGBoostRegressor model(/*n_estimators=*/50, /*learning_rate=*/0.3, /*max_depth=*/3,
                            /*lambda=*/1.0, /*gamma=*/0.0, /*min_child_weight=*/1.0);
    model.fit(X, y);

    auto preds = model.predict(X);
    REQUIRE(mse(preds, y) < 1.0);
}

TEST_CASE("Plus de rounds de boosting reduisent l'erreur d'entrainement", "[xgboost_regressor]") {
    Matrix X(6, 1);
    std::vector<double> y(6);
    for (std::size_t i = 0; i < 6; ++i) {
        X(i, 0) = static_cast<double>(i);
        y[i] = 2.0 * static_cast<double>(i) + 1.0;
    }

    XGBoostRegressor few_rounds(/*n_estimators=*/1, /*learning_rate=*/0.3, /*max_depth=*/3);
    few_rounds.fit(X, y);
    double mse_few = mse(few_rounds.predict(X), y);

    XGBoostRegressor many_rounds(/*n_estimators=*/50, /*learning_rate=*/0.3, /*max_depth=*/3);
    many_rounds.fit(X, y);
    double mse_many = mse(many_rounds.predict(X), y);

    REQUIRE(mse_many < mse_few);
}
