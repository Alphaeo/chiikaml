#include <catch2/catch_test_macros.hpp>

#include "chiikaml/matrix.hpp"

using chiikaml::Matrix;

TEST_CASE("Une matrice nouvellement creee a les bonnes dimensions", "[matrix]") {
    Matrix m(3, 4);
    REQUIRE(m.rows() == 3);
    REQUIRE(m.cols() == 4);
}

TEST_CASE("Les elements sont initialises a zero", "[matrix]") {
    Matrix m(2, 2);
    REQUIRE(m(0, 0) == 0.0);
    REQUIRE(m(1, 1) == 0.0);
    REQUIRE(m(0, 1) == 0.0);
    REQUIRE(m(1, 0) == 0.0);
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

TEST_CASE("Le produit matriciel refuse les dimensions incompatibles",
          "[matrix]") {
    Matrix a(2, 3);
    Matrix b(2, 4);

    REQUIRE_THROWS_AS(a * b, std::invalid_argument);
}

TEST_CASE("Le produit matriciel est correctement calcule", "[matrix]") {
    Matrix a(2, 2);
    Matrix b(2, 2);

    a(0, 0) = 1;
    a(0, 1) = 2;
    a(1, 0) = 3;
    a(1, 1) = 4;

    b(0, 0) = 5;
    b(0, 1) = 6;
    b(1, 0) = 7;
    b(1, 1) = 8;

    Matrix c = a * b;

    REQUIRE(c.rows() == 2);
    REQUIRE(c.cols() == 2);

    REQUIRE(c(0, 0) == 19); // 1*5 + 2*7
    REQUIRE(c(0, 1) == 22); // 1*6 + 2*8
    REQUIRE(c(1, 0) == 43); // 3*5 + 4*7
    REQUIRE(c(1, 1) == 50); // 3*6 + 4*8
}

TEST_CASE("resize agrandit et remplit les nouvelles cases a zero, en gardant les anciennes valeurs", "[matrix]") {
    Matrix m(2, 2);
    m(0, 0) = 1; m(0, 1) = 2;
    m(1, 0) = 3; m(1, 1) = 4;

    m.resize(3, 3);

    REQUIRE(m.rows() == 3);
    REQUIRE(m.cols() == 3);

    // Anciennes valeurs preservees
    REQUIRE(m(0, 0) == 1);
    REQUIRE(m(0, 1) == 2);
    REQUIRE(m(1, 0) == 3);
    REQUIRE(m(1, 1) == 4);

    // Nouvelles cases a zero
    REQUIRE(m(0, 2) == 0.0);
    REQUIRE(m(1, 2) == 0.0);
    REQUIRE(m(2, 0) == 0.0);
    REQUIRE(m(2, 1) == 0.0);
    REQUIRE(m(2, 2) == 0.0);
}

TEST_CASE("resize avec des dimensions asymetriques garde chaque valeur a la bonne case", "[matrix]") {
    // 2x3, chaque case a une valeur distincte : (i, j) -> i*10 + j
    Matrix m(2, 3);
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            m(i, j) = static_cast<double>(i * 10 + j);
        }
    }

    // Lignes qui augmentent (2 -> 3), colonnes qui diminuent (3 -> 2)
    // en meme temps : piege classique si on confond l'ancien et le
    // nouveau nombre de colonnes en recopiant les valeurs.
    m.resize(3, 2);

    REQUIRE(m.rows() == 3);
    REQUIRE(m.cols() == 2);

    REQUIRE(m(0, 0) == 0);   // etait (0,0) = 0*10+0
    REQUIRE(m(0, 1) == 1);   // etait (0,1) = 0*10+1
    REQUIRE(m(1, 0) == 10);  // etait (1,0) = 1*10+0
    REQUIRE(m(1, 1) == 11);  // etait (1,1) = 1*10+1

    // Nouvelle ligne, a zero
    REQUIRE(m(2, 0) == 0.0);
    REQUIRE(m(2, 1) == 0.0);
}

TEST_CASE("from_csv charge un fichier CSV valide", "[matrix]") {
    Matrix m = Matrix::from_csv(std::string(CHIIKAML_TEST_DATA_DIR) + "/sample.csv");

    REQUIRE(m.rows() == 2);
    REQUIRE(m.cols() == 3);
    REQUIRE(m(0, 0) == 1);
    REQUIRE(m(0, 1) == 2);
    REQUIRE(m(0, 2) == 3);
    REQUIRE(m(1, 0) == 4);
    REQUIRE(m(1, 1) == 5);
    REQUIRE(m(1, 2) == 6);
}

TEST_CASE("from_csv leve une exception si le fichier n'existe pas", "[matrix]") {
    REQUIRE_THROWS_AS(Matrix::from_csv(std::string(CHIIKAML_TEST_DATA_DIR) + "/does_not_exist.csv"),
                       std::runtime_error);
}

TEST_CASE("from_csv leve une exception si les lignes n'ont pas le meme nombre de colonnes", "[matrix]") {
    REQUIRE_THROWS_AS(Matrix::from_csv(std::string(CHIIKAML_TEST_DATA_DIR) + "/mismatched_columns.csv"),
                       std::invalid_argument);
}
