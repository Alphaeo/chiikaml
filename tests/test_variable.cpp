#include <catch2/catch_test_macros.hpp>

#include "chiikaml/variable.hpp"

using chiikaml::Tensor;
using chiikaml::Variable;

namespace {
Tensor scalar(double value) {
    Tensor t({1});
    t({0}) = value;
    return t;
}
} // namespace

TEST_CASE("Une Variable feuille garde sa valeur et un gradient initialise a zero", "[variable]") {
    Variable a(scalar(5.0), /*requires_grad=*/true);

    REQUIRE(a.data()({0}) == 5.0);
    REQUIRE(a.grad()({0}) == 0.0);
    REQUIRE(a.requires_grad());
}

TEST_CASE("accumulate_grad ajoute au gradient existant, sans l'ecraser", "[variable]") {
    Variable a(scalar(1.0), /*requires_grad=*/true);

    a.accumulate_grad(scalar(3.0));
    a.accumulate_grad(scalar(4.0));

    REQUIRE(a.grad()({0}) == 7.0);
}

TEST_CASE("accumulate_grad ne fait rien si requires_grad est faux", "[variable]") {
    Variable a(scalar(1.0), /*requires_grad=*/false);

    a.accumulate_grad(scalar(10.0));

    REQUIRE(a.grad()({0}) == 0.0);
}

TEST_CASE("zero_grad remet le gradient accumule a zero", "[variable]") {
    Variable a(scalar(1.0), /*requires_grad=*/true);
    a.accumulate_grad(scalar(9.0));

    a.zero_grad();

    REQUIRE(a.grad()({0}) == 0.0);
}

TEST_CASE("operator+ calcule la meme valeur que Tensor::operator+", "[variable]") {
    Variable a(scalar(2.0), /*requires_grad=*/true);
    Variable b(scalar(3.0), /*requires_grad=*/true);

    Variable c = a + b;

    REQUIRE(c.data()({0}) == 5.0);
}

TEST_CASE("backward() sur une Variable feuille seme simplement son propre gradient a 1", "[variable]") {
    Variable a(scalar(42.0), /*requires_grad=*/true);

    a.backward();

    REQUIRE(a.grad()({0}) == 1.0);
}

TEST_CASE("backward() sur une addition propage le gradient aux deux entrees", "[variable]") {
    Variable a(scalar(2.0), /*requires_grad=*/true);
    Variable b(scalar(3.0), /*requires_grad=*/true);

    Variable c = a + b;
    c.backward();

    // d(c)/da = 1, d(c)/db = 1
    REQUIRE(a.grad()({0}) == 1.0);
    REQUIRE(b.grad()({0}) == 1.0);
}

TEST_CASE("backward() accumule correctement en cas de fan-out (meme Variable utilisee deux fois)", "[variable]") {
    Variable a(scalar(5.0), /*requires_grad=*/true);

    Variable c = a + a;
    c.backward();

    // d(a+a)/da = 1 + 1 = 2 (les deux utilisations de a contribuent)
    REQUIRE(a.grad()({0}) == 2.0);
}

TEST_CASE("backward() se propage correctement a travers plusieurs niveaux du graphe", "[variable]") {
    Variable a(scalar(1.0), /*requires_grad=*/true);
    Variable b(scalar(2.0), /*requires_grad=*/true);
    Variable c(scalar(3.0), /*requires_grad=*/true);

    Variable d = (a + b) + c;
    d.backward();

    REQUIRE(a.grad()({0}) == 1.0);
    REQUIRE(b.grad()({0}) == 1.0);
    REQUIRE(c.grad()({0}) == 1.0);
}

TEST_CASE("backward() leve une exception si la Variable n'a pas exactement un element", "[variable]") {
    Variable a(Tensor({2, 2}), /*requires_grad=*/true);

    REQUIRE_THROWS_AS(a.backward(), std::invalid_argument);
}
