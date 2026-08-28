#pragma once

#include <cstddef>
#include <vector>
#include <algorithm>

#include "chiikaml/matrix.hpp"

namespace chiikaml {

// Classifieur k plus proches voisins (k-NN), version brute-force :
// pour predire un point, on calcule sa distance a *tous* les points
// d'entrainement, on garde les k plus proches, et on vote a la
// majorite parmi leurs labels.
//
// Complexite d'une prediction : O(n_train * n_features) pour les
// distances, puis O(n_train log k) pour trouver les k plus proches.
// Pas de structure d'index (ca viendra avec le KD-Tree en phase 3) :
// c'est volontairement la version "naive" de reference, celle a
// laquelle on comparera les futures optimisations.
class KNNClassifier {
public:
    // k = nombre de voisins consideres pour le vote. Doit etre >= 1.
    explicit KNNClassifier(std::size_t k);

    // Memorise le jeu d'entrainement. X : une ligne par echantillon,
    // une colonne par feature. y : un label entier par ligne de X.
    // X.rows() doit etre egal a y.size().
    void fit(const Matrix& X, std::vector<int> y);

    // Predit un label pour chaque ligne de X (chaque ligne = un point
    // a classer). Doit etre appelee apres fit().
    std::vector<int> predict(const Matrix& X) const;

private:
    std::size_t k_;
    Matrix X_train_;
    std::vector<int> y_train_;

    // Distance euclidienne au carre entre la ligne `query_row` de
    // `query` et la ligne `train_row` de X_train_.
    // (Au carre : pour comparer/trier des distances, pas besoin de
    // sqrt() qui est monotone et coute plus cher pour rien.)
    double squared_distance(const Matrix& query, std::size_t query_row,
                             std::size_t train_row) const;

    // Predit le label d'une seule ligne (query_row) de `query`.
    int predict_one(const Matrix& query, std::size_t query_row) const;
};

} // namespace chiikaml
