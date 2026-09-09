#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "chiikaml/matrix.hpp"
#include "chiikaml/xgboost_tree.hpp"

using chiikaml::Matrix;
using chiikaml::XGBoostTree;

TEST_CASE("XGBoostTree a profondeur 0 est une feuille unique, valeur = -G/(H+lambda)", "[xgboost_tree]") {
    Matrix X(2, 1);
    X(0, 0) = 1;
    X(1, 0) = 100; // valeurs tres differentes -- un split "gagnerait" normalement, mais max_depth l'interdit

    std::vector<double> gradients = {-2.0, -2.0};
    std::vector<double> hessians = {1.0, 1.0};

    XGBoostTree tree(/*lambda=*/1.0, /*gamma=*/0.0, /*max_depth=*/0, /*min_child_weight=*/0.0);
    tree.fit(X, gradients, hessians);

    // G = -4, H = 2, lambda = 1 -> w = -(-4)/(2+1) = 4/3
    auto preds = tree.predict(X);
    REQUIRE(preds[0] == Catch::Approx(4.0 / 3.0));
    REQUIRE(preds[1] == Catch::Approx(4.0 / 3.0));
}

TEST_CASE("XGBoostTree choisit une coupure qui separe nettement deux groupes de gradients", "[xgboost_tree]") {
    Matrix X(4, 1);
    X(0, 0) = 1;
    X(1, 0) = 2;
    X(2, 0) = 10;
    X(3, 0) = 11;

    std::vector<double> gradients = {-1.0, -1.0, 1.0, 1.0};
    std::vector<double> hessians = {1.0, 1.0, 1.0, 1.0};

    XGBoostTree tree(/*lambda=*/0.0, /*gamma=*/0.0, /*max_depth=*/3, /*min_child_weight=*/0.0);
    tree.fit(X, gradients, hessians);

    auto preds = tree.predict(X);

    // Groupe {1,2} : G=-2,H=2,lambda=0 -> w=1.0. Groupe {10,11} : G=2,H=2 -> w=-1.0.
    REQUIRE(preds[0] == Catch::Approx(1.0));
    REQUIRE(preds[1] == Catch::Approx(1.0));
    REQUIRE(preds[2] == Catch::Approx(-1.0));
    REQUIRE(preds[3] == Catch::Approx(-1.0));
}

TEST_CASE("Un gamma trop eleve empeche toute coupure (elagage)", "[xgboost_tree]") {
    Matrix X(4, 1);
    X(0, 0) = 1;
    X(1, 0) = 2;
    X(2, 0) = 10;
    X(3, 0) = 11;

    std::vector<double> gradients = {-1.0, -1.0, 1.0, 1.0};
    std::vector<double> hessians = {1.0, 1.0, 1.0, 1.0};

    // Meme donnees que le test precedent (le split y avait un gain
    // net de 2.0) -- gamma=100 doit le rendre non rentable.
    XGBoostTree tree(/*lambda=*/0.0, /*gamma=*/100.0, /*max_depth=*/3, /*min_child_weight=*/0.0);
    tree.fit(X, gradients, hessians);

    auto preds = tree.predict(X);

    // Aucune coupure ne depasse gamma=100 -> feuille unique. G=0,H=4,lambda=0 -> w=0.0
    for (double p : preds) {
        REQUIRE(p == Catch::Approx(0.0).margin(1e-12));
    }
}

TEST_CASE("min_child_weight rejette une coupure qui creerait un groupe trop leger", "[xgboost_tree]") {
    Matrix X(4, 1);
    X(0, 0) = 1;
    X(1, 0) = 2;
    X(2, 0) = 10;
    X(3, 0) = 11;

    std::vector<double> gradients = {-1.0, -1.0, 1.0, 1.0};
    std::vector<double> hessians = {0.1, 0.1, 0.1, 0.1}; // somme totale = 0.4

    // N'importe quel split cree un groupe avec au plus 0.2 de somme
    // de hessiens -- min_child_weight = 1.0 rejette donc TOUTE coupure.
    XGBoostTree tree(/*lambda=*/0.0, /*gamma=*/0.0, /*max_depth=*/3, /*min_child_weight=*/1.0);
    tree.fit(X, gradients, hessians);

    auto preds = tree.predict(X);
    for (double p : preds) {
        REQUIRE(p == Catch::Approx(0.0).margin(1e-12));
    }
}

TEST_CASE("max_depth limite la profondeur meme quand un split serait rentable", "[xgboost_tree]") {
    Matrix X(4, 1);
    X(0, 0) = 1;
    X(1, 0) = 2;
    X(2, 0) = 10;
    X(3, 0) = 11;

    std::vector<double> gradients = {-1.0, -1.0, 1.0, 1.0};
    std::vector<double> hessians = {1.0, 1.0, 1.0, 1.0};

    XGBoostTree tree(/*lambda=*/0.0, /*gamma=*/0.0, /*max_depth=*/0, /*min_child_weight=*/0.0);
    tree.fit(X, gradients, hessians);

    auto preds = tree.predict(X);
    // max_depth=0 -> feuille unique des la racine. G=0,H=4,lambda=0 -> w=0.0
    for (double p : preds) {
        REQUIRE(p == Catch::Approx(0.0).margin(1e-12));
    }
}
