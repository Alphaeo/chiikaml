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

TEST_CASE("transpose() inverse la forme", "[tensor]") {
    Tensor t({2, 3});

    Tensor transposed = t.transpose();

    REQUIRE(transposed.shape() == std::vector<std::size_t>{3, 2});
}

TEST_CASE("transpose() donne acces aux memes valeurs, indices echanges", "[tensor]") {
    // 2x3, chaque case a une valeur distincte : (i, j) -> i*10 + j
    Tensor t({2, 3});
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            t({i, j}) = static_cast<double>(i * 10 + j);
        }
    }

    Tensor transposed = t.transpose();

    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            REQUIRE(transposed({j, i}) == t({i, j}));
        }
    }
}

TEST_CASE("transpose() est une VUE : ecrire dedans modifie l'original", "[tensor]") {
    Tensor t({2, 3});
    Tensor transposed = t.transpose();

    // Ecrit dans la vue transposee...
    transposed({1, 0}) = 99.0;

    // ...et l'original doit refleter le changement, au bon endroit
    // (indices echanges : transposed(1,0) correspond a t(0,1)).
    REQUIRE(t({0, 1}) == 99.0);
}

TEST_CASE("reshape() garde le meme nombre d'elements et change la forme", "[tensor]") {
    Tensor t({2, 6});

    Tensor reshaped = t.reshape({3, 4});

    REQUIRE(reshaped.shape() == std::vector<std::size_t>{3, 4});
    REQUIRE(reshaped.size() == t.size());
}

TEST_CASE("reshape() leve une exception si le nombre total d'elements ne correspond pas", "[tensor]") {
    Tensor t({2, 3}); // 6 elements

    REQUIRE_THROWS_AS(t.reshape({2, 2}), std::invalid_argument); // 4 elements
}

TEST_CASE("reshape() preserve l'ordre des valeurs (parcours a plat identique)", "[tensor]") {
    // 2x3, rempli 0..5 dans l'ordre du parcours -- (i,j) -> i*3 + j
    Tensor t({2, 3});
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            t({i, j}) = static_cast<double>(i * 3 + j);
        }
    }

    Tensor reshaped = t.reshape({3, 2});

    // Meme parcours a plat (0,1,2,3,4,5), juste regroupe en 3x2 au
    // lieu de 2x3.
    REQUIRE(reshaped({0, 0}) == 0);
    REQUIRE(reshaped({0, 1}) == 1);
    REQUIRE(reshaped({1, 0}) == 2);
    REQUIRE(reshaped({1, 1}) == 3);
    REQUIRE(reshaped({2, 0}) == 4);
    REQUIRE(reshaped({2, 1}) == 5);
}

TEST_CASE("reshape() est une VUE : ecrire dedans modifie l'original", "[tensor]") {
    Tensor t({2, 3});
    Tensor reshaped = t.reshape({3, 2});

    reshaped({2, 1}) = 77.0; // position plate 5 -> t(1, 2)

    REQUIRE(t({1, 2}) == 77.0);
}

TEST_CASE("broadcast_to ajoute des dimensions implicites a gauche", "[tensor]") {
    // Vecteur de 3 valeurs distinctes, etire vers {2, 3} : les deux
    // "lignes" doivent lire les 3 memes valeurs.
    Tensor v({3});
    v({0}) = 10; v({1}) = 20; v({2}) = 30;

    Tensor broadcasted = v.broadcast_to({2, 3});

    REQUIRE(broadcasted.shape() == std::vector<std::size_t>{2, 3});
    for (std::size_t j = 0; j < 3; ++j) {
        REQUIRE(broadcasted({0, j}) == v({j}));
        REQUIRE(broadcasted({1, j}) == v({j}));
    }
}

TEST_CASE("broadcast_to etire une dimension de taille 1", "[tensor]") {
    Tensor t({1, 3});
    t({0, 0}) = 1; t({0, 1}) = 2; t({0, 2}) = 3;

    Tensor broadcasted = t.broadcast_to({4, 3});

    REQUIRE(broadcasted.shape() == std::vector<std::size_t>{4, 3});
    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE(broadcasted({i, 0}) == 1);
        REQUIRE(broadcasted({i, 1}) == 2);
        REQUIRE(broadcasted({i, 2}) == 3);
    }
}

TEST_CASE("broadcast_to leve une exception si les dimensions sont incompatibles", "[tensor]") {
    Tensor t({2}); // ni egal a 3, ni egal a 1

    REQUIRE_THROWS_AS(t.broadcast_to({4, 3}), std::invalid_argument);
}

TEST_CASE("broadcast_to leve une exception si target_shape a moins de dimensions", "[tensor]") {
    Tensor t({2, 3});

    REQUIRE_THROWS_AS(t.broadcast_to({3}), std::invalid_argument);
}
