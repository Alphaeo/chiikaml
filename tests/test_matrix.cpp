#include <catch2/catch_test_macros.hpp>

#include "cppml/matrix.hpp"

using cppml::Matrix;

TEST_CASE("Une matrice nouvellement creee a les bonnes dimensions", "[matrix]") {
    Matrix m(3, 4);
    REQUIRE(m.rows() == 3);
    REQUIRE(m.cols() == 4);
}

TEST_CASE("Les elements sont initialises a zero", "[matrix]") {
    Matrix m(2, 2);
    REQUIRE(m(0, 0) == 0.0);
    REQUIRE(m(1, 1) == 0.0);
}

TEST_CASE("On peut lire et ecrire un element via operator()", "[matrix]") {
    Matrix m(2, 2);
    m(0, 1) = 42.0;
    REQUIRE(m(0, 1) == 42.0);
    REQUIRE(m(1, 0) == 0.0); // les autres cases ne doivent pas bouger
}

TEST_CASE("L'addition de deux matrices se fait element par element", "[matrix]") {
    Matrix a(2, 2);
    Matrix b(2, 2);
    a(0, 0) = 1;
    a(0, 1) = 2;
    a(1, 0) = 3;
    a(1, 1) = 4;

    b(0, 0) = 10;
    b(0, 1) = 20;
    b(1, 0) = 30;
    b(1, 1) = 40;

    Matrix c = a + b;

    REQUIRE(c(0, 0) == 11);
    REQUIRE(c(0, 1) == 22);
    REQUIRE(c(1, 0) == 33);
    REQUIRE(c(1, 1) == 44);
}
