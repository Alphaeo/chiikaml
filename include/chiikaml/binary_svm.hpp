#pragma once

#include <cstddef>
#include <vector>

#include "chiikaml/matrix.hpp"
#include "chiikaml/svm_kernels.hpp"

namespace chiikaml {

// Classifieur SVM binaire interne.
//
// Cette classe contient le veritable algorithme SVM. Elle accepte
// uniquement des labels encodes avec :
//
//     -1 : classe negative
//     +1 : classe positive
//
// Elle est utilisee par SVMClassifier, qui se charge de transformer
// les labels originaux et de gerer la classification multiclasse.
//
// Le modele est entraine dans sa formulation duale :
//
//     f(x) = sum(alpha_i * y_i * K(x_i, x)) + intercept
//
// Les observations dont alpha_i est strictement positif sont les
// vecteurs supports.
class BinarySVM {
public:
    // C controle la penalisation des violations de marge et doit etre
    // strictement positif.
    //
    // gamma est utilise par les kernels Polynomial et RBF. Une valeur
    // de 0 demande une selection automatique :
    //
    //     gamma = 1 / number_of_features
    //
    // degree et coef0 sont utilises uniquement par le kernel
    // Polynomial.
    //
    // max_iterations fixe le nombre maximal d'iterations de SMO.
    //
    // tolerance controle les conditions d'arret et la detection des
    // violations des conditions KKT.
    explicit BinarySVM(
        double C = 1.0,
        SVMKernel kernel = SVMKernel::RBF,
        double gamma = 0.0,
        std::size_t degree = 3,
        double coef0 = 0.0,
        std::size_t max_iterations = 1000,
        double tolerance = 1e-4,
        bool fit_intercept = true,
        unsigned int seed = 42
    );

    // Entraine le SVM binaire.
    //
    // X contient une ligne par echantillon et une colonne par feature.
    // y doit contenir exactement une valeur -1 ou +1 par ligne de X.
    //
    // Les deux classes doivent etre presentes.
    void fit(const Matrix& X, const std::vector<int>& y);

    // Renvoie -1 ou +1 pour chaque ligne de X.
    //
    // Doit etre appelee apres fit().
    std::vector<int> predict(const Matrix& X) const;

    // Renvoie le score signe de chaque ligne de X.
    //
    // score >= 0 correspond a la classe positive.
    // score < 0 correspond a la classe negative.
    std::vector<double> decision_function(
        const Matrix& X
    ) const;

    // Renvoie le score signe d'une seule ligne.
    //
    // Cette fonction est principalement utilisee par le wrapper
    // multiclasse SVMClassifier.
    double decision_value(
        const Matrix& X,
        std::size_t sample_row
    ) const;

    // Renvoie les vecteurs supports.
    const Matrix& support_vectors() const;

    // Renvoie les coefficients duals signes :
    //
    //     alpha_i * y_i
    //
    // dans le meme ordre que les lignes de support_vectors().
    const std::vector<double>& dual_coefficients() const;

    // Renvoie l'intercept appris.
    double intercept() const;

    double C() const;
    SVMKernel kernel() const;

    // Renvoie la valeur effective de gamma.
    //
    // Si gamma = 0 a ete demande, cette valeur est calculee pendant
    // fit() a partir du nombre de features.
    double gamma() const;

    std::size_t degree() const;
    double coef0() const;

    bool converged() const;
    std::size_t iterations() const;
    std::size_t number_of_support_vectors() const;

private:
    double C_;
    SVMKernel kernel_;

    // Value requested by the user. Zero means automatic gamma.
    double configured_gamma_;

    // Gamma actually used after fit().
    double gamma_;

    std::size_t degree_;
    double coef0_;
    std::size_t max_iterations_;
    double tolerance_;
    bool fit_intercept_;
    unsigned int seed_;

    bool fitted_;
    bool converged_;
    std::size_t iterations_;
    std::size_t n_features_;

    // Only samples with a nonzero alpha are retained after training.
    Matrix support_vectors_;

    // Signed coefficients alpha_i * y_i corresponding to the rows
    // of support_vectors_.
    std::vector<double> dual_coefficients_;

    double intercept_;

    // Evaluates the selected kernel between two matrix rows.
    double evaluate_kernel(
        const Matrix& first,
        std::size_t first_row,
        const Matrix& second,
        std::size_t second_row
    ) const;

    // Verifies that the model is fitted and that X has the expected
    // number of features.
    void validate_prediction_data(const Matrix& X) const;
};

} // namespace chiikaml::detail