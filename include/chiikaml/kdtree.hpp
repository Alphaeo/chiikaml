#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "chiikaml/matrix.hpp"

namespace chiikaml {

// Index spatial pour accelerer la recherche des k plus proches
// voisins, comparee a l'approche brute-force de KNNClassifier qui
// compare a *tous* les points d'entrainement a chaque requete
// (O(n)). Un KD-Tree partitionne l'espace recursivement (en
// alternant la dimension de decoupe a chaque niveau), ce qui permet
// d'elaguer des branches entieres de la recherche sans les visiter.
// Complexite moyenne d'une requete : O(log n) au lieu de O(n) --
// mais ca depend fortement du nombre de dimensions (le fameau
// "fleau de la dimension" : au-dela d'une grosse dizaine de
// dimensions, l'elagage devient inefficace et on retombe pres du
// brute-force).
//
// Volontairement independant de KNNClassifier : KDTree ne connait
// que la geometrie (des points, des indices), pas les labels. Le
// lien entre les deux (utiliser KDTree comme "backend" rapide pour
// KNNClassifier) sera fait dans une etape ulterieure.
class KDTree {
public:
    // Construit l'index a partir de tous les points de `points`
    // (une ligne = un point). La construction reorganise les points
    // en arbre une bonne fois pour toutes ; les requetes ulterieures
    // n'ont plus besoin de reparcourir tout le jeu de donnees.
    explicit KDTree(const Matrix& points);

    // Renvoie les indices (dans `points`, celle passee au
    // constructeur) des k plus proches voisins de la ligne
    // `query_row` de `query`, tries du plus proche au plus loin.
    std::vector<std::size_t> nearest_neighbors(const Matrix& query, std::size_t query_row,
                                                std::size_t k) const;

private:
    // Un noeud de l'arbre : un point de reference (son indice dans
    // points_), la dimension selon laquelle ce noeud "coupe"
    // l'espace en deux (alternee a chaque niveau de profondeur), et
    // ses deux sous-arbres. unique_ptr : chaque noeud est possede
    // par exactement un parent (ou par KDTree pour la racine) --
    // quand le parent est detruit, ses enfants le sont
    // automatiquement, recursivement, sans fuite memoire et sans
    // gestion manuelle de delete.
    struct Node {
        std::size_t point_index;
        std::size_t split_dim;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;

        Node(std::size_t point_index_, std::size_t split_dim_)
            : point_index(point_index_), split_dim(split_dim_) {}
    };

    Matrix points_;
    std::unique_ptr<Node> root_;

    // Construit recursivement le sous-arbre couvrant
    // indices[lo, hi) (borne haute exclue), a la profondeur `depth`
    // (qui determine la dimension de decoupe : depth % points_.cols()).
    // Partitionne `indices` sur place (nth_element) : c'est pour ca
    // que `indices` est une reference non-const.
    std::unique_ptr<Node> build(std::vector<std::size_t>& indices, std::size_t lo, std::size_t hi,
                                 std::size_t depth);

    // Parcourt recursivement le sous-arbre `node`, en maintenant
    // dans `best` les k meilleurs candidats trouves jusqu'ici
    // (tries par distance croissante). Modifie `best` en place.
    void search(const Node* node, const Matrix& query, std::size_t query_row, std::size_t k,
                std::vector<std::pair<double, std::size_t>>& best) const;

    double squared_distance(const Matrix& query, std::size_t query_row,
                             std::size_t point_index) const;
};

} // namespace chiikaml
