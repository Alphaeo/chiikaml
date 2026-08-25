#include "chiikaml/kdtree.hpp"

#include <algorithm>
#include <stdexcept>

namespace chiikaml {

// TODO(toi):
// - construit un vector<size_t> indices = {0, 1, ..., points.rows()-1}
// - stocke points dans points_
// - appelle build(indices, 0, indices.size(), 0) et stocke le
//   resultat dans root_
KDTree::KDTree(const Matrix& points)
    : points_(0, 0) {
    std::vector<std::size_t> indices(points.rows());
    for (std::size_t i=0; i<points.rows(); ++i) {
        indices[i] = i;
    }
    points_ = points;
    root_ = build(indices, 0, indices.size(), 0);
}

// TODO(toi), etape par etape :
// - cas de base : si lo == hi, il n'y a rien a construire, renvoie
//   un unique_ptr<Node> vide (nullptr implicite -- un unique_ptr par
//   defaut ne possede rien)
// - split_dim = depth % points_.cols()
// - mid = lo + (hi - lo) / 2
// - std::nth_element sur indices[lo, hi) autour de indices[mid],
//   avec un comparateur qui compare points_(a, split_dim) et
//   points_(b, split_dim) pour deux indices a, b
// - cree le noeud : auto node = std::make_unique<Node>(indices[mid], split_dim);
// - node->left  = build(indices, lo, mid, depth + 1);
// - node->right = build(indices, mid + 1, hi, depth + 1);
// - renvoie node (un unique_ptr se deplace automatiquement au retour,
//   pas besoin de std::move ici)
std::unique_ptr<KDTree::Node> KDTree::build(std::vector<std::size_t>& indices, std::size_t lo,
                                             std::size_t hi, std::size_t depth) {
    if (lo >= hi) {
        return nullptr;
    }
    std::size_t split_dim = depth % points_.cols();
    std::size_t mid = lo + (hi - lo) / 2;
    std::nth_element(indices.begin() + lo, indices.begin() + mid, indices.begin() + hi,
                     [&split_dim, this](std::size_t a, std::size_t b) {
                         return points_(a, split_dim) < points_(b, split_dim);
                     });
    auto node = std::make_unique<Node>(indices[mid], split_dim);
    node->left = build(indices, lo, mid, depth + 1);
    node->right = build(indices, mid + 1, hi, depth + 1);
    return node;
}

// TODO(toi), etape par etape :
// - si node == nullptr, rien a faire, return
// - calcule dist = squared_distance(query, query_row, node->point_index)
// - insere (dist, node->point_index) dans `best` a la bonne position
//   pour garder `best` trie par distance croissante, puis si
//   best.size() > k, retire le dernier (le plus mauvais)
//   (regarde std::lower_bound pour trouver la position d'insertion,
//   ou fais-le "a la main" avec une boucle -- les deux sont valides)
// - diff = query(query_row, node->split_dim) - points_(node->point_index, node->split_dim)
// - decide quel sous-arbre visiter en premier selon le signe de diff
//   (si diff < 0, le point requete est du cote "left" de la coupe)
// - visite ce sous-arbre en premier (appel recursif a search)
// - elagage : ne visite l'AUTRE sous-arbre que si best.size() < k,
//   OU si diff * diff < best.back().first (la pire distance
//   actuellement gardee) -- sinon cette branche ne peut
//   mathematiquement contenir aucun point plus proche que ce qu'on
//   a deja trouve, pas la peine de la visiter
void KDTree::search(const Node* node, const Matrix& query, std::size_t query_row, std::size_t k,
                     std::vector<std::pair<double, std::size_t>>& best) const {
    if (node == nullptr){
        return;
    }
    double dist = squared_distance(query, query_row, node->point_index);
    auto it = std::lower_bound(best.begin(), best.end(), std::make_pair(dist, node->point_index),
                               [](const std::pair<double, std::size_t>& a,
                                  const std::pair<double, std::size_t>& b) {
                                   return a.first < b.first;
                               });
    best.insert(it, std::make_pair(dist, node->point_index));
    if (best.size() > k) {
        best.pop_back();
    }
    double diff = query(query_row, node->split_dim) - points_(node->point_index, node->split_dim);
    if (diff < 0) {
    search(node->left.get(), query, query_row, k, best);
    if (best.size() < k || diff * diff < best.back().first) {
        search(node->right.get(), query, query_row, k, best);  // l'AUTRE côté
        }
    } else {
        search(node->right.get(), query, query_row, k, best);
    if (best.size() < k || diff * diff < best.back().first) {
        search(node->left.get(), query, query_row, k, best);   // l'AUTRE côté
        }
    }
    return;
}

// TODO(toi): identique a KNNClassifier::squared_distance, mais par
// rapport a points_ au lieu de X_train_.
double KDTree::squared_distance(const Matrix& query, std::size_t query_row,
                                 std::size_t point_index) const {
    double squared_distance = 0;
    for (std::size_t f=0; f<points_.cols(); ++f){
        double diff = query(query_row, f) - points_(point_index, f);
        squared_distance += diff * diff;
    }
    return squared_distance;
}

// TODO(toi):
// - best : vector<pair<double,size_t>> vide
// - appelle search(root_.get(), query, query_row, k, best)
// - construit et renvoie un vector<size_t> a partir des .second de
//   `best` (qui est deja trie par distance croissante)
std::vector<std::size_t> KDTree::nearest_neighbors(const Matrix& query, std::size_t query_row,
                                                     std::size_t k) const {
    std::vector<std::pair<double, std::size_t>> best;
    search(root_.get(), query, query_row, k, best);
    std::vector<std::size_t> indices;
    indices.reserve(best.size());
    for (const auto& pair : best) {
        indices.push_back(pair.second);
    }
    return indices;
}

} // namespace chiikaml
