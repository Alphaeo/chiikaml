#include "chiikaml/kmeans.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace chiikaml {

// TODO(toi):
// - stocke n_clusters et max_iterations
// - construit rng_ a partir de seed (regarde le constructeur de
//   std::mt19937)
// - centroids_ : pas encore de vraies valeurs disponibles (X n'est
//   pas encore connu) -- comme pour X_train_/points_ avant, il faut
//   quand meme l'initialiser a quelque chose dans la liste
//   d'initialisation
KMeans::KMeans(std::size_t n_clusters, std::size_t max_iterations, unsigned int seed)
    : centroids_(0, 0) {
    throw std::logic_error("KMeans::KMeans pas encore implemente");
}

// TODO(toi): identique au squared_distance des modules precedents,
// mais entre deux points qui peuvent venir de deux matrices
// differentes (a et b). a.cols() et b.cols() doivent etre egaux
// (pas verifie ici pour rester simple, mais garde-le en tete).
double KMeans::squared_distance(const Matrix& a, std::size_t row_a, const Matrix& b,
                                 std::size_t row_b) {
    throw std::logic_error("KMeans::squared_distance pas encore implemente");
}

// TODO(toi): calcule squared_distance entre la ligne `row` de
// `points` et chaque ligne de centroids_ (n_clusters_ d'entre
// elles), renvoie l'indice de la plus proche.
std::size_t KMeans::nearest_centroid(const Matrix& points, std::size_t row) const {
    throw std::logic_error("KMeans::nearest_centroid pas encore implemente");
}

// TODO(toi), etape par etape :
//
// 1) Initialisation des centres :
//    - construit un vector<size_t> indices = {0, 1, ..., X.rows()-1}
//      (regarde std::iota)
//    - tire n_clusters_ indices distincts au sort parmi eux, avec
//      rng_ (regarde std::sample -- il ecrit son resultat via un
//      iterateur de sortie, par exemple std::back_inserter sur un
//      vector<size_t> vide au depart)
//    - centroids_ = Matrix(n_clusters_, X.cols()), et pour chaque
//      centre i, copie la ligne indices_tires[i] de X dedans
//    - labels_ : vector<size_t> de taille X.rows(), rempli de 0
//
// 2) Boucle principale, jusqu'a max_iterations_ fois :
//    a) affectation : pour chaque ligne i de X, calcule
//       nearest_centroid(X, i). Si different de labels_[i], met a
//       jour labels_[i] et note que "quelque chose a change" ce tour
//    b) mise a jour : recalcule chaque centre comme la moyenne des
//       points qui lui sont actuellement assignes (parcours X une
//       fois, accumule une somme par cluster + un compteur de
//       points par cluster, puis divise) -- attention a la division
//       par zero si un cluster n'a recupere aucun point (cas limite,
//       pas teste pour l'instant, mais bon reflexe d'y penser)
//    c) si rien n'a change a l'etape (a), la solution a converge :
//       sors de la boucle avant d'atteindre max_iterations_
void KMeans::fit(const Matrix& X) {
    throw std::logic_error("KMeans::fit pas encore implemente");
}

// TODO(toi): pour chaque ligne de X, appelle nearest_centroid et
// empile le resultat -- tres proche de KNNClassifier::predict().
std::vector<std::size_t> KMeans::predict(const Matrix& X) const {
    throw std::logic_error("KMeans::predict pas encore implemente");
}

const std::vector<std::size_t>& KMeans::labels() const {
    throw std::logic_error("KMeans::labels pas encore implemente");
}

const Matrix& KMeans::centroids() const {
    throw std::logic_error("KMeans::centroids pas encore implemente");
}

} // namespace chiikaml
