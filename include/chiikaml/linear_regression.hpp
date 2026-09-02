#pragma once

#include <cstddef>
#include <vector>

#include "chiikaml/matrix.hpp"

namespace chiikaml {

// Regression lineaire par la methode des moindres carres.
//
// Le modele cherche une relation lineaire entre les features X et
// la variable cible y :
//
//     y_pred = X * coefficients + intercept
//
// L'entrainement cherche les coefficients et l'intercept qui
// minimisent la somme des erreurs au carre :
//
//     sum((y_true - y_pred)^2)
class LinearRegression {
public:
    // Si fit_intercept vaut true, le modele apprend une constante
    // en plus des coefficients associes aux features.
    //
    // Si fit_intercept vaut false, l'intercept reste egal a 0 et
    // le modele est force a passer par l'origine.
    explicit LinearRegression(bool fit_intercept = true);

    // Entraine le modele.
    //
    // X : une ligne par echantillon, une colonne par feature.
    // y : une valeur cible par ligne de X.
    //
    // X.rows() doit etre egal a y.size().
    // X et y ne doivent pas etre vides.
    void fit(const Matrix& X, const std::vector<double>& y);

    // Predit une valeur pour chaque ligne de X.
    //
    // X doit avoir le meme nombre de colonnes que le jeu
    // d'entrainement. Doit etre appelee apres fit().
    std::vector<double> predict(const Matrix& X) const;

    // Renvoie les coefficients appris, dans le meme ordre que les
    // colonnes du jeu d'entrainement.
    const std::vector<double>& coefficients() const;

    // Renvoie l'intercept appris.
    //
    // Vaut toujours 0 si le modele a ete construit avec
    // fit_intercept = false.
    double intercept() const;

private:
    bool fit_intercept_;
    bool fitted_;
    std::size_t n_features_;

    std::vector<double> coefficients_;
    double intercept_;

    // Predit la valeur correspondant a une seule ligne de X.
    double predict_one(const Matrix& X,
                       std::size_t sample_row) const;
};

} // namespace chiikaml