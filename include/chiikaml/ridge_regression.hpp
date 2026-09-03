#pragma once

#include <cstddef>
#include <vector>

#include "chiikaml/matrix.hpp"

namespace chiikaml {

// Regression lineaire avec regularisation L2 (Ridge).
//
// Le modele cherche une relation lineaire entre les features X et
// la variable cible y :
//
//     y_pred = X * coefficients + intercept
//
// Contrairement a la regression lineaire ordinaire, Ridge ajoute une
// penalisation sur la taille des coefficients :
//
//     sum((y_true - y_pred)^2)
//         + alpha * sum(coefficients[j]^2)
//
// Cette regularisation limite les coefficients trop grands, reduit le
// surapprentissage et stabilise le modele lorsque des features sont
// fortement correlees.
//
// L'intercept n'est pas penalise.
class RidgeRegression {
public:
    // alpha controle la force de la regularisation et doit etre
    // positif ou nul :
    //
    // - alpha = 0 : equivalent a une regression lineaire ordinaire ;
    // - alpha faible : regularisation legere ;
    // - alpha eleve : coefficients davantage rapproches de zero.
    //
    // Si fit_intercept vaut true, le modele apprend une constante
    // en plus des coefficients associes aux features.
    explicit RidgeRegression(
        double alpha = 1.0,
        bool fit_intercept = true
    );

    // Entraine le modele.
    //
    // X : une ligne par echantillon, une colonne par feature.
    // y : une valeur cible par ligne de X.
    //
    // X.rows() doit etre egal a y.size().
    // X et y ne doivent pas etre vides.
    //
    // L'entrainement resout le systeme :
    //
    //     (X^T X + alpha * I) * coefficients = X^T y
    //
    // en utilisant une decomposition de Cholesky.
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

    // Renvoie la force de regularisation utilisee par le modele.
    double alpha() const;

private:
    double alpha_;
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