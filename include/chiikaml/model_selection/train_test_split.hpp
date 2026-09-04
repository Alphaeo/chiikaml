#pragma once

#include <vector>

#include "chiikaml/matrix.hpp"

namespace chiikaml::model_selection {

// Resultat d'une separation entre jeu d'entrainement et jeu de test.
//
// Target vaut generalement :
// - int pour une classification ;
// - double pour une regression.

// Exemple d'utilisation :

// auto split =
//     chiikaml::model_selection::train_test_split(
//         X,
//         y,
//         0.2
//     );

// Matrix X_train = split.X_train;
// Matrix X_test = split.X_test;

// std::vector<int> y_train = split.y_train;
// std::vector<int> y_test = split.y_test;
template<typename Target>
struct TrainTestSplitResult {
    Matrix X_train;
    Matrix X_test;
    std::vector<Target> y_train;
    std::vector<Target> y_test;
};

// Separe un jeu de donnees en un jeu d'entrainement et un jeu de test.
//
// X contient une ligne par echantillon.
// y contient une cible par ligne de X.
//
// test_size est la proportion d'echantillons placee dans le jeu de
// test et doit etre strictement comprise entre 0 et 1.
//
// Si shuffle vaut true, les lignes sont melangees avant la separation.
// seed permet de reproduire exactement le meme melange.
template<typename Target>
TrainTestSplitResult<Target> train_test_split(
    const Matrix& X,
    const std::vector<Target>& y,
    double train_size = 0.2,
    bool shuffle = true,
    unsigned int seed = 42
);

} // namespace chiikaml::model_selection