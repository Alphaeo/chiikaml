#pragma once

#include <cstddef>
#include <vector>

#include "chiikaml/matrix.hpp"

namespace chiikaml {

// Regression lineaire avec regularisation L1 (Lasso).
//
// Le modele cherche une relation lineaire entre les features X et
// la variable cible y :
//
//     y_pred = X * coefficients + intercept
//
// Lasso minimise la fonction objectif :
//
//     0.5 * sum((y_true - y_pred)^2)
//         + alpha * sum(abs(coefficients[j]))
//
// La regularisation L1 peut rendre certains coefficients exactement
// nuls. Lasso peut donc effectuer une selection automatique des
// features.
//
// L'intercept n'est pas penalise.
class LassoRegression {
public:
    // alpha controle la force de la regularisation et doit etre
    // positif ou nul.
    //
    // max_iterations fixe le nombre maximal de passages complets
    // sur les coefficients.
    //
    // tolerance controle le critere d'arret de l'algorithme.
    explicit LassoRegression(
        double alpha = 1.0,
        bool fit_intercept = true,
        std::size_t max_iterations = 1000,
        double tolerance = 1e-6
    );

    // Entraine le modele par descente par coordonnees.
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
    // Vaut toujours 0 si fit_intercept est false.
    double intercept() const;

    // Renvoie la force de regularisation.
    double alpha() const;

    // Indique si le critere de convergence a ete atteint avant le
    // nombre maximal d'iterations.
    bool converged() const;

    // Renvoie le nombre d'iterations effectuees pendant le dernier
    // entrainement.
    std::size_t iterations() const;

private:
    double alpha_;
    bool fit_intercept_;
    std::size_t max_iterations_;
    double tolerance_;

    bool fitted_;
    bool converged_;
    std::size_t iterations_;
    std::size_t n_features_;

    std::vector<double> coefficients_;
    double intercept_;

    // Predit la valeur correspondant a une seule ligne de X.
    double predict_one(const Matrix& X,
                       std::size_t sample_row) const;
};

} // namespace chiikaml