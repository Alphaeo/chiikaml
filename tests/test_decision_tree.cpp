#include <catch2/catch_test_macros.hpp>

#include "chiikaml/decision_tree.hpp"
#include "chiikaml/matrix.hpp"

using chiikaml::DecisionTreeClassifier;
using chiikaml::Matrix;

namespace {

// Une seule feature, deux groupes bien separes : {1,2,3} -> classe 0,
// {8,9,10} -> classe 1.
Matrix make_1d_points() {
    Matrix X(6, 1);
    X(0, 0) = 1.0;
    X(1, 0) = 2.0;
    X(2, 0) = 3.0;
    X(3, 0) = 8.0;
    X(4, 0) = 9.0;
    X(5, 0) = 10.0;
    return X;
}

std::vector<int> make_1d_labels() {
    return {0, 0, 0, 1, 1, 1};
}

// Deux clusters bien separes en 2D (memes points que test_knn.cpp).
Matrix make_2d_points() {
    Matrix X(6, 2);
    X(0, 0) = 0.0;  X(0, 1) = 0.0;
    X(1, 0) = 0.0;  X(1, 1) = 1.0;
    X(2, 0) = 1.0;  X(2, 1) = 0.0;
    X(3, 0) = 10.0; X(3, 1) = 10.0;
    X(4, 0) = 10.0; X(4, 1) = 11.0;
    X(5, 0) = 11.0; X(5, 1) = 10.0;
    return X;
}

std::vector<int> make_2d_labels() {
    return {0, 0, 0, 1, 1, 1};
}

} // namespace

TEST_CASE("Un arbre de decision separe deux groupes 1D bien distincts", "[decision_tree]") {
    DecisionTreeClassifier tree;
    tree.fit(make_1d_points(), make_1d_labels());

    Matrix query(2, 1);
    query(0, 0) = 2.5;
    query(1, 0) = 8.5;

    auto predictions = tree.predict(query);

    REQUIRE(predictions.size() == 2);
    REQUIRE(predictions[0] == 0);
    REQUIRE(predictions[1] == 1);
}

TEST_CASE("Un arbre de decision separe deux clusters 2D bien distincts", "[decision_tree]") {
    DecisionTreeClassifier tree;
    tree.fit(make_2d_points(), make_2d_labels());

    Matrix query(2, 2);
    query(0, 0) = 0.5;  query(0, 1) = 0.5;
    query(1, 0) = 10.5; query(1, 1) = 10.5;

    auto predictions = tree.predict(query);

    REQUIRE(predictions[0] == 0);
    REQUIRE(predictions[1] == 1);
}

TEST_CASE("Sans contrainte de profondeur, l'arbre retrouve les labels d'entrainement", "[decision_tree]") {
    Matrix X = make_2d_points();
    std::vector<int> y = make_2d_labels();

    DecisionTreeClassifier tree;
    tree.fit(X, y);

    auto predictions = tree.predict(X);

    REQUIRE(predictions == y);
}

TEST_CASE("max_depth limite la profondeur sans faire planter l'arbre", "[decision_tree]") {
    Matrix X = make_2d_points();
    std::vector<int> y = make_2d_labels();

    DecisionTreeClassifier tree(1, 2);
    tree.fit(X, y);

    auto predictions = tree.predict(X);

    REQUIRE(predictions.size() == 6);
    for (int label : predictions) {
        REQUIRE((label == 0 || label == 1));
    }
}
