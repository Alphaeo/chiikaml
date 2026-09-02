#include <catch2/catch_test_macros.hpp>

#include "chiikaml/tensor.hpp"

using chiikaml::Tensor;

TEST_CASE("Un tenseur nouvellement cree a la bonne forme et taille", "[tensor]") {
    Tensor t({2, 3, 4});

    REQUIRE(t.ndim() == 3);
    REQUIRE(t.shape() == std::vector<std::size_t>{2, 3, 4});
    REQUIRE(t.size() == 24);
}

TEST_CASE("Les elements sont initialises a zero", "[tensor]") {
    Tensor t({2, 3, 4});

    REQUIRE(t({0, 0, 0}) == 0.0);
    REQUIRE(t({1, 2, 3}) == 0.0);
}

TEST_CASE("On peut lire et ecrire un element via operator(), sans toucher les autres", "[tensor]") {
    // 3 dimensions (pas juste 2, comme Matrix) : verifie que les
    // strides sont bien calcules pour un tenseur qui n'a PAS
    // d'equivalent Matrix.
    Tensor t({2, 3, 4});

    t({1, 2, 3}) = 42.0;

    REQUIRE(t({1, 2, 3}) == 42.0);
    // Des voisins proches dans les indices, mais qui doivent rester
    // a zero -- piege classique si les strides sont mal calcules.
    REQUIRE(t({1, 2, 2}) == 0.0);
    REQUIRE(t({1, 1, 3}) == 0.0);
    REQUIRE(t({0, 2, 3}) == 0.0);
}

TEST_CASE("L'addition de deux tenseurs de meme forme se fait element par element", "[tensor]") {
    Tensor a({2, 2});
    Tensor b({2, 2});
    a({0, 0}) = 1;  a({0, 1}) = 2;  a({1, 0}) = 3;  a({1, 1}) = 4;
    b({0, 0}) = 10; b({0, 1}) = 20; b({1, 0}) = 30; b({1, 1}) = 40;

    Tensor c = a + b;

    REQUIRE(c({0, 0}) == 11);
    REQUIRE(c({0, 1}) == 22);
    REQUIRE(c({1, 0}) == 33);
    REQUIRE(c({1, 1}) == 44);
}

TEST_CASE("L'addition leve une exception si les formes ne correspondent pas", "[tensor]") {
    Tensor a({2, 2});
    Tensor b({2, 3});

    REQUIRE_THROWS_AS(a + b, std::invalid_argument);
}
