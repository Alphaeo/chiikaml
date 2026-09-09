#pragma once

#include <cstddef>
#include <vector>

#include "chiikaml/matrix.hpp"
#include "chiikaml/xgboost_tree.hpp"

namespace chiikaml {

// Gradient boosting regularise (l'algorithme XGBoost), pour la
// regression (perte quadratique 0.5*(y-yhat)^2). A chaque round m
// (parmi n_estimators) :
//
//  1. calcule, pour chaque exemple, le gradient g_i et le hessien h_i
//     de la perte par rapport a la prediction ACTUELLE du modele.
//     Pour la perte quadratique : g_i = yhat_i - y_i (la derivee),
//     h_i = 1 (la derivee seconde, constante -- ce qui fait de la
//     regression le cas le plus simple pour valider tout le pipeline
//     avant d'attaquer des pertes plus complexes comme la log-loss
//     pour la classification, ou h_i varie par exemple).
//
//  2. entraine un XGBoostTree sur (X, gradients, hessiens) -- PAS sur
//     (X, y) directement.
//
//  3. met a jour les predictions : yhat_i += learning_rate *
//     arbre.predict(x_i). Le learning_rate ("shrinkage") reduit la
//     contribution de chaque arbre ; un pas trop grand a chaque round
//     sur-apprendrait vite et mal, un petit pas laisse aux rounds
//     suivants la place d'affiner encore.
//
// Difference fondamentale avec RandomForestClassifier : la ou
// RandomForest entraine n_trees arbres INDEPENDANTS (chacun sur son
// propre echantillon bootstrap, "embarrassingly parallel", un thread
// par arbre), XGBoost entraine ses arbres SEQUENTIELLEMENT -- chaque
// nouvel arbre depend des predictions accumulees par TOUS les arbres
// precedents (il apprend a corriger LEURS erreurs residuelles).
// Impossible de paralleliser l'ensemble arbre par arbre pour cette
// raison. La vraie piste de parallelisation pour XGBoost -- remise a
// une passe de perf ulterieure, comme le SIMD sur Matrix/Tensor --
// c'est a l'INTERIEUR de chaque arbre : paralleliser
// XGBoostTree::find_best_split() sur les features (chercher le
// meilleur split de plusieurs features en meme temps, avec le meme
// thread-pool que RandomForest) plutot que sur les arbres.
class XGBoostRegressor {
public:
    // n_estimators : nombre de rounds de boosting (= nombre d'arbres).
    // learning_rate : facteur de reduction applique a la contribution
    //   de chaque arbre (voir le commentaire de classe).
    // max_depth, lambda, gamma, min_child_weight : transmis tels
    //   quels a chaque XGBoostTree (voir son constructeur pour le
    //   sens de chacun).
    explicit XGBoostRegressor(std::size_t n_estimators = 100, double learning_rate = 0.1,
                               std::size_t max_depth = 3, double lambda = 1.0, double gamma = 0.0,
                               double min_child_weight = 1.0);

    // Entraine le modele : n_estimators rounds de boosting successifs.
    void fit(const Matrix& X, const std::vector<double>& y);

    // Predit une valeur pour chaque ligne de X. Doit etre appelee
    // apres fit().
    std::vector<double> predict(const Matrix& X) const;

private:
    std::size_t n_estimators_;
    double learning_rate_;
    std::size_t max_depth_;
    double lambda_;
    double gamma_;
    double min_child_weight_;

    // Prediction de depart, avant le premier arbre -- la moyenne de y
    // (le meilleur "modele constant" possible pour la perte
    // quadratique, avant meme de regarder les features).
    double base_prediction_ = 0.0;

    // Meme remarque que RandomForestClassifier::trees_ : XGBoostTree
    // contient un unique_ptr<Node>, donc n'est pas copiable -- chaque
    // arbre doit etre deplace (std::move) ou construit directement a
    // sa place dans le vector, jamais copie.
    std::vector<XGBoostTree> trees_;
};

} // namespace chiikaml
