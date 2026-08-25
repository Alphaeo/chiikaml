#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "chiikaml/kmeans.hpp"
#include "chiikaml/matrix.hpp"

using Catch::Approx;
using chiikaml::KMeans;
using chiikaml::Matrix;

namespace {

// Deux clusters bien separes en 2D (memes points que test_knn.cpp) :
// - 3 points autour de (0, 0) : indices 0, 1, 2
// - 3 points autour de (10, 10) : indices 3, 4, 5
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

} // namespace

TEST_CASE("K-Means separe deux clusters bien distincts", "[kmeans]") {
    Matrix X = make_points();
    KMeans km(2);
    km.fit(X);

    const auto& labels = km.labels();
    REQUIRE(labels.size() == 6);

    // Les 3 premiers points doivent finir dans le meme cluster...
    REQUIRE(labels[0] == labels[1]);
    REQUIRE(labels[1] == labels[2]);
    // ...les 3 derniers aussi...
    REQUIRE(labels[3] == labels[4]);
    REQUIRE(labels[4] == labels[5]);
    // ...mais les deux groupes n'ont pas le meme label.
    REQUIRE(labels[0] != labels[3]);
}

TEST_CASE("Les centres trouves sont proches des vrais centres des clusters", "[kmeans]") {
    Matrix X = make_points();
    KMeans km(2);
    km.fit(X);

    std::size_t cluster0 = km.labels()[0];
    std::size_t cluster1 = km.labels()[3];

    // Moyenne de (0,0), (0,1), (1,0) = (1/3, 1/3)
    REQUIRE(km.centroids()(cluster0, 0) == Approx(1.0 / 3.0).margin(0.01));
    REQUIRE(km.centroids()(cluster0, 1) == Approx(1.0 / 3.0).margin(0.01));

    // Moyenne de (10,10), (10,11), (11,10) = (31/3, 31/3)
    REQUIRE(km.centroids()(cluster1, 0) == Approx(31.0 / 3.0).margin(0.01));
    REQUIRE(km.centroids()(cluster1, 1) == Approx(31.0 / 3.0).margin(0.01));
}

TEST_CASE("predict() classe un nouveau point sans modifier les centres", "[kmeans]") {
    Matrix X = make_points();
    KMeans km(2);
    km.fit(X);

    Matrix centroids_before = km.centroids();

    Matrix query(1, 2);
    query(0, 0) = 0.2;
    query(0, 1) = 0.2;

    auto predictions = km.predict(query);

    REQUIRE(predictions.size() == 1);
    REQUIRE(predictions[0] == km.labels()[0]);

    for (std::size_t i = 0; i < centroids_before.rows(); ++i) {
        for (std::size_t j = 0; j < centroids_before.cols(); ++j) {
            REQUIRE(km.centroids()(i, j) == centroids_before(i, j));
        }
    }
}

TEST_CASE("Le meme seed donne un resultat deterministe", "[kmeans]") {
    Matrix X = make_points();

    KMeans km1(2, 100, 7);
    km1.fit(X);

    KMeans km2(2, 100, 7);
    km2.fit(X);

    REQUIRE(km1.labels() == km2.labels());
}
