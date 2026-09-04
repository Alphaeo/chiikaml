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

TEST_CASE("matmul calcule le produit matriciel classique", "[tensor]") {
    // a (2x3) * b (3x2) = c (2x2)
    Tensor a({2, 3});
    a({0, 0}) = 1; a({0, 1}) = 2; a({0, 2}) = 3;
    a({1, 0}) = 4; a({1, 1}) = 5; a({1, 2}) = 6;

    Tensor b({3, 2});
    b({0, 0}) = 7;  b({0, 1}) = 8;
    b({1, 0}) = 9;  b({1, 1}) = 10;
    b({2, 0}) = 11; b({2, 1}) = 12;

    Tensor c = a.matmul(b);

    REQUIRE(c.shape() == std::vector<std::size_t>{2, 2});
    // c(0,0) = 1*7 + 2*9 + 3*11 = 58
    REQUIRE(c({0, 0}) == 58);
    // c(0,1) = 1*8 + 2*10 + 3*12 = 64
    REQUIRE(c({0, 1}) == 64);
    // c(1,0) = 4*7 + 5*9 + 6*11 = 139
    REQUIRE(c({1, 0}) == 139);
    // c(1,1) = 4*8 + 5*10 + 6*12 = 154
    REQUIRE(c({1, 1}) == 154);
}

TEST_CASE("matmul leve une exception si les dimensions internes ne correspondent pas", "[tensor]") {
    Tensor a({2, 3});
    Tensor b({4, 2}); // 4 != 3

    REQUIRE_THROWS_AS(a.matmul(b), std::invalid_argument);
}

TEST_CASE("matmul en lot (batch) calcule chaque matrice de la pile independamment", "[tensor]") {
    // a et b ont tous les deux un batch de taille 2 (aucun
    // broadcasting necessaire) : deux 2x2 * 2x2 independants.
    Tensor a({2, 2, 2});
    // batch 0 : [[1,2],[3,4]]
    a({0, 0, 0}) = 1; a({0, 0, 1}) = 2;
    a({0, 1, 0}) = 3; a({0, 1, 1}) = 4;
    // batch 1 : [[5,6],[7,8]]
    a({1, 0, 0}) = 5; a({1, 0, 1}) = 6;
    a({1, 1, 0}) = 7; a({1, 1, 1}) = 8;

    Tensor b({2, 2, 2});
    // batch 0 : identite
    b({0, 0, 0}) = 1; b({0, 0, 1}) = 0;
    b({0, 1, 0}) = 0; b({0, 1, 1}) = 1;
    // batch 1 : permutation (echange les colonnes)
    b({1, 0, 0}) = 0; b({1, 0, 1}) = 1;
    b({1, 1, 0}) = 1; b({1, 1, 1}) = 0;

    Tensor c = a.matmul(b);

    REQUIRE(c.shape() == std::vector<std::size_t>{2, 2, 2});
    // batch 0 : a0 * identite = a0
    REQUIRE(c({0, 0, 0}) == 1); REQUIRE(c({0, 0, 1}) == 2);
    REQUIRE(c({0, 1, 0}) == 3); REQUIRE(c({0, 1, 1}) == 4);
    // batch 1 : a1 * permutation = colonnes de a1 echangees
    REQUIRE(c({1, 0, 0}) == 6); REQUIRE(c({1, 0, 1}) == 5);
    REQUIRE(c({1, 1, 0}) == 8); REQUIRE(c({1, 1, 1}) == 7);
}

TEST_CASE("matmul broadcast un tenseur sans dimension de lot sur l'autre", "[tensor]") {
    // a a un batch de 2, b n'en a pas (2D pur) -- b doit etre
    // "reutilise" pour les deux matrices du lot de a.
    Tensor a({2, 2, 3});
    a({0, 0, 0}) = 1; a({0, 0, 1}) = 2; a({0, 0, 2}) = 3;
    a({0, 1, 0}) = 4; a({0, 1, 1}) = 5; a({0, 1, 2}) = 6;
    a({1, 0, 0}) = 7;  a({1, 0, 1}) = 8;  a({1, 0, 2}) = 9;
    a({1, 1, 0}) = 10; a({1, 1, 1}) = 11; a({1, 1, 2}) = 12;

    Tensor b({3, 2});
    b({0, 0}) = 1; b({0, 1}) = 0;
    b({1, 0}) = 0; b({1, 1}) = 1;
    b({2, 0}) = 1; b({2, 1}) = 1;

    Tensor c = a.matmul(b);

    REQUIRE(c.shape() == std::vector<std::size_t>{2, 2, 2});
    REQUIRE(c({0, 0, 0}) == 4);  REQUIRE(c({0, 0, 1}) == 5);
    REQUIRE(c({0, 1, 0}) == 10); REQUIRE(c({0, 1, 1}) == 11);
    REQUIRE(c({1, 0, 0}) == 16); REQUIRE(c({1, 0, 1}) == 17);
    REQUIRE(c({1, 1, 0}) == 22); REQUIRE(c({1, 1, 1}) == 23);
}

TEST_CASE("matmul leve une exception si les dimensions de lot ne sont pas broadcast-compatibles", "[tensor]") {
    Tensor a({2, 2, 2}); // batch = 2
    Tensor b({3, 2, 2}); // batch = 3 -- ni egal, ni 1

    REQUIRE_THROWS_AS(a.matmul(b), std::invalid_argument);
}

TEST_CASE("matmul leve une exception si un des deux tenseurs a moins de 2 dimensions", "[tensor]") {
    Tensor a({3});    // 1D, pas de partie matricielle
    Tensor b({3, 2});

    REQUIRE_THROWS_AS(a.matmul(b), std::invalid_argument);
}
