#include <catch2/catch_test_macros.hpp>

#include "chiikaml/knn.hpp"
#include "chiikaml/matrix.hpp"

using chiikaml::KNNClassifier;
using chiikaml::Matrix;

namespace {

// Deux clusters bien separes en 2D :
// - 3 points autour de (0, 0), label 0
// - 3 points autour de (10, 10), label 1
Matrix make_training_points() {
    Matrix X(6, 2);
    X(0, 0) = 0.0;  X(0, 1) = 0.0;
    X(1, 0) = 0.0;  X(1, 1) = 1.0;
    X(2, 0) = 1.0;  X(2, 1) = 0.0;
    X(3, 0) = 10.0; X(3, 1) = 10.0;
    X(4, 0) = 10.0; X(4, 1) = 11.0;
    X(5, 0) = 11.0; X(5, 1) = 10.0;
    return X;
}

std::vector<int> make_training_labels() {
    return {0, 0, 0, 1, 1, 1};
}

} // namespace

TEST_CASE("k-NN classe un point proche du cluster 0 comme label 0", "[knn]") {
    KNNClassifier knn(3);
    knn.fit(make_training_points(), make_training_labels());

    Matrix query(1, 2);
    query(0, 0) = 0.5;
    query(0, 1) = 0.5;

    auto predictions = knn.predict(query);

    REQUIRE(predictions.size() == 1);
    REQUIRE(predictions[0] == 0);
}

TEST_CASE("k-NN classe un point proche du cluster 1 comme label 1", "[knn]") {
    KNNClassifier knn(3);
    knn.fit(make_training_points(), make_training_labels());

    Matrix query(1, 2);
    query(0, 0) = 10.5;
    query(0, 1) = 10.5;

    auto predictions = knn.predict(query);

    REQUIRE(predictions.size() == 1);
    REQUIRE(predictions[0] == 1);
}

TEST_CASE("k-NN peut predire plusieurs points en un seul appel", "[knn]") {
    KNNClassifier knn(1);
    knn.fit(make_training_points(), make_training_labels());

    Matrix query(2, 2);
    query(0, 0) = 0.1;  query(0, 1) = 0.1;
    query(1, 0) = 10.9; query(1, 1) = 10.9;

    auto predictions = knn.predict(query);

    REQUIRE(predictions.size() == 2);
    REQUIRE(predictions[0] == 0);
    REQUIRE(predictions[1] == 1);
}

TEST_CASE("avec k=1, un point d'entrainement se retrouve lui-meme", "[knn]") {
    KNNClassifier knn(1);
    Matrix X = make_training_points();
    knn.fit(X, make_training_labels());

    // Reutilise directement un point d'entrainement comme requete :
    // sa distance a lui-meme est 0, donc c'est forcement son plus
    // proche voisin.
    Matrix query(1, 2);
    query(0, 0) = 10.0;
    query(0, 1) = 10.0;

    auto predictions = knn.predict(query);

    REQUIRE(predictions[0] == 1);
}
