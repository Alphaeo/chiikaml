#include <algorithm>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "chiikaml/kdtree.hpp"
#include "chiikaml/matrix.hpp"

using chiikaml::KDTree;
using chiikaml::Matrix;

namespace {

// Deux clusters bien separes en 2D (mêmes points que test_knn.cpp) :
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

// Reference independante, volontairement naive (brute-force), pour
// verifier le KDTree par comparaison plutot que par calcul a la main.
std::vector<std::size_t> brute_force_nearest(const Matrix& points, const Matrix& query,
                                              std::size_t query_row, std::size_t k) {
    std::vector<std::pair<double, std::size_t>> distances;
    distances.reserve(points.rows());
    for (std::size_t i = 0; i < points.rows(); ++i) {
        double squared = 0.0;
        for (std::size_t f = 0; f < points.cols(); ++f) {
            double diff = query(query_row, f) - points(i, f);
            squared += diff * diff;
        }
        distances.push_back({squared, i});
    }
    std::sort(distances.begin(), distances.end());

    std::vector<std::size_t> result;
    for (std::size_t i = 0; i < k && i < distances.size(); ++i) {
        result.push_back(distances[i].second);
    }
    return result;
}

Matrix random_points(std::size_t rows, std::size_t cols, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    Matrix m(rows, cols);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            m(i, j) = dist(rng);
        }
    }
    return m;
}

} // namespace

TEST_CASE("Un point identique a un point d'entrainement est son propre plus proche voisin", "[kdtree]") {
    Matrix points = make_points();
    KDTree tree(points);

    Matrix query(1, 2);
    query(0, 0) = 10.0;
    query(0, 1) = 10.0;

    auto neighbors = tree.nearest_neighbors(query, 0, 1);

    REQUIRE(neighbors.size() == 1);
    REQUIRE(neighbors[0] == 3);
}

TEST_CASE("Les k plus proches voisins sont renvoyes tries par distance croissante", "[kdtree]") {
    Matrix points = make_points();
    KDTree tree(points);

    Matrix query(1, 2);
    query(0, 0) = 0.2;
    query(0, 1) = 0.05;

    auto neighbors = tree.nearest_neighbors(query, 0, 3);

    REQUIRE(neighbors == std::vector<std::size_t>{0, 2, 1});
}

TEST_CASE("KDTree est d'accord avec une recherche brute-force sur des donnees aleatoires", "[kdtree]") {
    std::mt19937 rng(42); // seed fixe : test reproductible

    Matrix points = random_points(200, 3, rng);
    KDTree tree(points);

    Matrix queries = random_points(20, 3, rng);

    for (std::size_t q = 0; q < queries.rows(); ++q) {
        auto expected = brute_force_nearest(points, queries, q, 5);
        auto actual = tree.nearest_neighbors(queries, q, 5);
        REQUIRE(actual == expected);
    }
}
