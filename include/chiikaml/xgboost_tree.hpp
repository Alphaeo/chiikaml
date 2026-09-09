#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "chiikaml/matrix.hpp"

namespace chiikaml {

// Arbre de regression utilise EN INTERNE par XGBoostRegressor, un par
// round de boosting. Difference cle avec DecisionTreeClassifier :
//
//  - il ne voit JAMAIS les vraies etiquettes y : il est entraine sur
//    le GRADIENT (g_i) et le HESSIEN (h_i) de la fonction de perte,
//    calcules par XGBoostRegressor a partir des predictions ACTUELLES
//    du modele -- une approximation de Newton du second ordre de "de
//    combien, et dans quelle direction, faudrait-il bouger la
//    prediction de chaque exemple pour reduire la perte".
//
//  - il ne choisit pas ses coupures par Gini/entropie, mais par le
//    "gain" XGBoost -- pour une coupure qui separe un groupe (sommes
//    G, H de ses gradients/hessiens) en gauche (GL, HL) et droite
//    (GR, HR) :
//
//      gain = 0.5 * [ GL^2/(HL+lambda) + GR^2/(HR+lambda)
//                      - (GL+GR)^2/(HL+HR+lambda) ] - gamma
//
//    Les deux premiers termes mesurent le gain de "separer" (la perte
//    diminue d'autant si on coupe ici plutot que de garder un seul
//    groupe) ; gamma est une penalite FIXE par coupure -- une coupure
//    n'est gardee que si son gain reste positif APRES cette penalite
//    ("elagage" automatique, pas de coupure qui complexifie l'arbre
//    pour un gain marginal).
//
//  - la valeur d'une feuille n'est pas une moyenne/majorite, mais la
//    solution EXACTE (en une etape, pas de descente iterative) du
//    minimum de la perte regularisee sur cette feuille :
//    w* = -G / (H + lambda). lambda (regularisation L2) adoucit cette
//    valeur vers 0 quand H est petit (peu de "poids" -- peu confiant
//    dans cette feuille), l'empechant de sur-reagir a une poignee
//    d'exemples.
//
// Meme choix de simplification volontaire que DecisionTreeClassifier
// (v1 correcte et lisible d'abord, optimisable plus tard) : chaque
// appel recursif travaille sur une COPIE du sous-ensemble qui lui
// revient, plutot que sur des indices dans un tableau partage.
class XGBoostTree {
public:
    // lambda : regularisation L2 sur les poids de feuille (>= 0).
    // gamma : gain minimum requis pour garder une coupure (>= 0),
    //   voir le commentaire de classe pour la formule complete.
    // max_depth : profondeur maximale de l'arbre.
    // min_child_weight : somme minimale de hessiens requise dans
    //   CHAQUE sous-groupe resultant d'une coupure -- une autre forme
    //   de regularisation XGBoost, qui empeche de creer des feuilles
    //   "sur-specifiques" a une poignee d'exemples (mesuree en somme
    //   de hessiens, pas en nombre de points : un point avec un
    //   hessien tres petit "pese" moins dans cette contrainte).
    explicit XGBoostTree(double lambda = 1.0, double gamma = 0.0, std::size_t max_depth = 3,
                          double min_child_weight = 1.0);

    // Construit l'arbre a partir de X et des gradients/hessiens
    // PRECALCULES par XGBoostRegressor (un g_i et un h_i par ligne de
    // X, meme ordre -- pas les vraies etiquettes y).
    void fit(const Matrix& X, const std::vector<double>& gradients, const std::vector<double>& hessians);

    // Renvoie, pour chaque ligne de X, la valeur de la feuille dans
    // laquelle elle tombe (PAS encore multipliee par le learning
    // rate -- XGBoostRegressor s'en charge, cet arbre ne connait pas
    // le learning rate).
    std::vector<double> predict(const Matrix& X) const;

private:
    struct Node {
        bool is_leaf = true;

        // Valide seulement si is_leaf == true.
        double leaf_value = 0.0;

        // Valides seulement si is_leaf == false.
        std::size_t split_feature = 0;
        double split_threshold = 0.0;
        std::unique_ptr<Node> left;   // X(split_feature) <= split_threshold
        std::unique_ptr<Node> right;  // X(split_feature) >  split_threshold
    };

    double lambda_;
    double gamma_;
    std::size_t max_depth_;
    double min_child_weight_;
    std::unique_ptr<Node> root_;

    // Poids optimal d'une feuille dont les exemples ont pour sommes
    // de gradients/hessiens G et H : w* = -G / (H + lambda). Ne
    // depend d'aucun etat de l'arbre autre que lambda -- static,
    // comme DecisionTreeClassifier::gini_from_counts.
    static double leaf_weight(double G, double H, double lambda);

    // Cherche, parmi toutes les features de X et tous les seuils
    // candidats, la coupure qui maximise le gain XGBoost (voir le
    // commentaire de classe pour la formule). Ecrit le resultat dans
    // best_feature/best_threshold/best_gain ; renvoie false si aucune
    // coupure valide n'a ete trouvee (gain <= 0 partout, ou toutes
    // rejetees par min_child_weight_).
    //
    // Meme technique que DecisionTreeClassifier::find_best_split :
    // pour chaque feature, trie les points par valeur croissante,
    // puis balaie cet ordre de gauche a droite en deplacant un point
    // a la fois du groupe "droite" vers "gauche", en maintenant des
    // sommes GL/HL de proche en proche (O(1) par point deplace)
    // plutot que de tout resommer a chaque seuil candidat (O(n) par
    // candidat). Contrairement a find_best_split de
    // DecisionTreeClassifier, celle-ci n'est PAS static : le calcul
    // du gain depend de lambda_/gamma_, et le filtrage des candidats
    // depend de min_child_weight_.
    bool find_best_split(const Matrix& X, const std::vector<double>& gradients,
                          const std::vector<double>& hessians, std::size_t& best_feature,
                          double& best_threshold, double& best_gain) const;

    // Construit recursivement le sous-arbre pour le sous-ensemble
    // (X, gradients, hessians) deja filtre pour ce noeud, a la
    // profondeur `depth` (0 = racine).
    std::unique_ptr<Node> build(const Matrix& X, const std::vector<double>& gradients,
                                 const std::vector<double>& hessians, std::size_t depth) const;

    double predict_one(const Matrix& X, std::size_t row) const;
};

} // namespace chiikaml
