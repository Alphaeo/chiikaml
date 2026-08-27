// Demo bout-en-bout du dashboard : entraine un KMeans, l'affiche
// dans le navigateur, puis pousse une deuxieme version quelques
// secondes plus tard SUR LA MEME PAGE (pas de nouvel onglet) --
// exactement le comportement "premier appel ouvre, appels suivants
// mettent a jour" qu'on voulait.
#include <chrono>
#include <iostream>
#include <thread>

#include "chiikaml/kmeans.hpp"
#include "chiikaml/matrix.hpp"
#include "chiikaml/viz/visualize_kmeans.hpp"

using chiikaml::KMeans;
using chiikaml::Matrix;

namespace {

Matrix two_clusters() {
    Matrix X(6, 2);
    X(0, 0) = 0.0;  X(0, 1) = 0.0;
    X(1, 0) = 0.0;  X(1, 1) = 1.0;
    X(2, 0) = 1.0;  X(2, 1) = 0.0;
    X(3, 0) = 10.0; X(3, 1) = 10.0;
    X(4, 0) = 10.0; X(4, 1) = 11.0;
    X(5, 0) = 11.0; X(5, 1) = 10.0;
    return X;
}

Matrix three_clusters() {
    Matrix X(9, 2);
    X(0, 0) = 0.0;  X(0, 1) = 0.0;
    X(1, 0) = 0.0;  X(1, 1) = 1.0;
    X(2, 0) = 1.0;  X(2, 1) = 0.0;
    X(3, 0) = 10.0; X(3, 1) = 10.0;
    X(4, 0) = 10.0; X(4, 1) = 11.0;
    X(5, 0) = 11.0; X(5, 1) = 10.0;
    X(6, 0) = 0.0;  X(6, 1) = 10.0;
    X(7, 0) = 1.0;  X(7, 1) = 11.0;
    X(8, 0) = 0.0;  X(8, 1) = 11.0;
    return X;
}

} // namespace

int main() {
    std::cout << "Entrainement KMeans (2 clusters) et ouverture du dashboard...\n";
    Matrix X1 = two_clusters();
    KMeans km1(2);
    km1.fit(X1);
    chiikaml::viz::visualize(km1, X1);

    std::cout << "Dashboard ouvert. Mise a jour en direct dans 5 secondes (meme page, pas de nouvel onglet)...\n";
    std::this_thread::sleep_for(std::chrono::seconds(5));

    Matrix X2 = three_clusters();
    KMeans km2(3);
    km2.fit(X2);
    chiikaml::viz::visualize(km2, X2);

    std::cout << "Mis a jour (3 clusters). Le programme reste actif 60 secondes pour que tu puisses regarder.\n";
    std::this_thread::sleep_for(std::chrono::seconds(60));

    return 0;
}
