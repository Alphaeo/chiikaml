#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "chiikaml/matrix.hpp"

namespace chiikaml {

// Arbre de decision de classification (CART : Classification And
// Regression Tree), construit par decoupes binaires recursives.
//
// A chaque noeud interne, on choisit UNE feature et UN seuil qui
// separent les points en deux groupes (X(feature) <= seuil a
// gauche, X(feature) > seuil a droite) de maniere a rendre chaque
// groupe le plus "pur" possible -- c'est-a-dire dominé par une
// seule classe. On mesure l'impurete d'un groupe avec l'indice de
// Gini :
//
//   gini(y) = 1 - somme_c (p_c)^2
//
// ou p_c est la proportion de la classe c dans y. gini = 0 quand le
// groupe est parfaitement pur (une seule classe), et augmente vers
// 1 quand les classes sont melangees. Pour choisir un split, on
// teste plusieurs (feature, seuil) candidats et on garde celui qui
// minimise la moyenne ponderee des gini des deux groupes resultants
// (ponderee par leur taille respective).
//
// La recursion s'arrete (le noeud devient une feuille, qui predit
// la classe majoritaire du groupe qui y arrive) quand :
//  - le groupe est deja pur (gini == 0), ou
//  - la profondeur max (max_depth_) est atteinte, ou
//  - le groupe est trop petit pour continuer a le diviser
//    (min_samples_split_), ou
//  - aucun split ne separe mieux les points (ex : toutes les
//    valeurs identiques sur toutes les features).
//
// Choix de simplification volontaire pour cette premiere version :
// chaque appel recursif travaille sur une COPIE du sous-ensemble de
// points qui lui revient (un sous-Matrix/sous-vector<int>), plutot
// que sur des indices dans un tableau partage (comme KDTree::build
// le faisait avec nth_element). C'est moins efficace (copies), mais
// beaucoup plus simple a lire et raisonner -- optimisable plus tard
// (Phase 7).
class DecisionTreeClassifier {
public:
    explicit DecisionTreeClassifier(std::size_t max_depth = 5, std::size_t min_samples_split = 2);

    // Entraine l'arbre sur (X, y). Une ligne de X = un point, une
    // entree de y = son label.
    void fit(const Matrix& X, std::vector<int> y);

    // Predit un label pour chaque ligne de X. Doit etre appelee
    // apres fit().
    std::vector<int> predict(const Matrix& X) const;

private:
    struct Node {
        // Un noeud est soit une feuille (is_leaf == true), soit un
        // noeud de decision interne. Design volontairement simple
        // (un seul struct avec un indicateur), plutot que deux types
        // distincts lies par heritage/polymorphisme -- une
        // alternative valable a explorer plus tard si besoin.
        bool is_leaf = true;

        // Valide seulement si is_leaf == true.
        int predicted_class = 0;

        // Valides seulement si is_leaf == false.
        std::size_t split_feature = 0;
        double split_threshold = 0.0;
        std::unique_ptr<Node> left;   // X(split_feature) <= split_threshold
        std::unique_ptr<Node> right;  // X(split_feature) >  split_threshold
    };

    std::size_t max_depth_;
    std::size_t min_samples_split_;
    std::unique_ptr<Node> root_;

    // Construit recursivement le sous-arbre pour le sous-ensemble
    // (X, y) deja filtre pour ce noeud, a la profondeur `depth`
    // (0 = racine).
    std::unique_ptr<Node> build(const Matrix& X, const std::vector<int>& y, std::size_t depth) const;

    // Impurete de Gini de l'ensemble de labels y. Ne depend d'aucun
    // etat de l'arbre -- static, comme KMeans::squared_distance.
    static double gini(const std::vector<int>& y);

    // La classe la plus frequente dans y (utilisee pour la
    // prediction d'une feuille).
    static int majority_class(const std::vector<int>& y);

    // Cherche, parmi toutes les features de X et plusieurs seuils
    // candidats par feature, le split qui minimise l'impurete
    // ponderee des deux groupes resultants. Ecrit le resultat dans
    // best_feature/best_threshold ; renvoie false si aucun split
    // valide n'a ete trouve (ex: toutes les valeurs identiques).
    static bool find_best_split(const Matrix& X, const std::vector<int>& y, std::size_t& best_feature,
                                 double& best_threshold);

    int predict_one(const Matrix& X, std::size_t row) const;
};

} // namespace chiikaml
