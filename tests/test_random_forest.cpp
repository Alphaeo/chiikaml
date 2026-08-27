#include <catch2/catch_test_macros.hpp>

#include "chiikaml/matrix.hpp"
#include "chiikaml/random_forest.hpp"

using chiikaml::Matrix;
using chiikaml::RandomForestClassifier;

namespace {

// Deux clusters bien separes en 2D (memes points que test_knn.cpp).
Matrix make_points() {
    Matrix X(6, 2);
    X(0, 0) = 0.0;  X(0, 1) = 0.0;
    X(1, 0) = 0.0;  X(1, 1) = 1.0;
    X(2, 0) = 1.0;  X(2, 1) = 0.0;
    X(3, 0) = 10.0; X(3, 1) = 10.0;
    X(4, 0) = 10.0; X(4, 1) = 11.0;
    X(5, 0) = 11.0; X(5, 1) = 10.0;
    return X;
}

std::vector<int> make_labels() {
    return {0, 0, 0, 1, 1, 1};
}

} // namespace

TEST_CASE("Une foret separe deux clusters bien distincts", "[random_forest]") {
    RandomForestClassifier forest(10, 5, 2, 42);
    forest.fit(make_points(), make_labels());

    Matrix query(2, 2);
    query(0, 0) = 0.5;  query(0, 1) = 0.5;
    query(1, 0) = 10.5; query(1, 1) = 10.5;

    auto predictions = forest.predict(query);

    REQUIRE(predictions.size() == 2);
    REQUIRE(predictions[0] == 0);
    REQUIRE(predictions[1] == 1);
}

TEST_CASE("La foret retrouve les labels d'entrainement sur des clusters bien separes", "[random_forest]") {
    Matrix X = make_points();
    std::vector<int> y = make_labels();

    RandomForestClassifier forest(10, 5, 2, 42);
    forest.fit(X, y);

    auto predictions = forest.predict(X);

    REQUIRE(predictions == y);
}

TEST_CASE("Le meme seed donne un resultat deterministe", "[random_forest]") {
    Matrix X = make_points();
    std::vector<int> y = make_labels();

    RandomForestClassifier forest1(10, 5, 2, 7);
    forest1.fit(X, y);

    RandomForestClassifier forest2(10, 5, 2, 7);
    forest2.fit(X, y);

    REQUIRE(forest1.predict(X) == forest2.predict(X));
}
