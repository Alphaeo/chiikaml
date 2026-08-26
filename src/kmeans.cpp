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
    max_iterations_ = max_iterations;
    n_clusters_ = n_clusters;
    rng_ = std::mt19937(seed);
    centroids_ = Matrix(n_clusters_, 0);
}

// TODO(toi): identique au squared_distance des modules precedents,
// mais entre deux points qui peuvent venir de deux matrices
// differentes (a et b). a.cols() et b.cols() doivent etre egaux
// (pas verifie ici pour rester simple, mais garde-le en tete).
double KMeans::squared_distance(const Matrix& a, std::size_t row_a, const Matrix& b,
                                 std::size_t row_b) {
    double squared_distance = 0;
    if (a.cols() != b.cols()) {
        throw std::invalid_argument("Matrices a et b doivent avoir le meme nombre de colonnes");
    }
    for (std::size_t j = 0; j < a.cols(); ++j) {
        double diff = a(row_a, j) - b(row_b, j);
        squared_distance += diff * diff;
    }
    return squared_distance;
}

// TODO(toi): calcule squared_distance entre la ligne `row` de
// `points` et chaque ligne de centroids_ (n_clusters_ d'entre
// elles), renvoie l'indice de la plus proche.
std::size_t KMeans::nearest_centroid(const Matrix& points, std::size_t row) const {
    std::size_t nearest_index = 0;
    double min_distance = squared_distance(points, row, centroids_, 0);
    for (std::size_t i = 1; i < n_clusters_; ++i){
        double distance = squared_distance(points, row, centroids_, i);
        if (distance < min_distance) {
            min_distance = distance;
            nearest_index = i;
        }
    }
    return nearest_index;
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
    std::vector<std::size_t> indices(X.rows());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<std::size_t> sampled_indices;
    std::sample(indices.begin(), indices.end(), std::back_inserter(sampled_indices), n_clusters_, rng_);
    centroids_.resize(n_clusters_, X.cols());

    for (std::size_t i = 0; i < n_clusters_; ++i) {
        for(std::size_t j = 0; j < X.cols(); ++j) {
            centroids_(i, j) = X(sampled_indices[i], j);
        }
    }

    std::vector<std::size_t> new_labels(X.rows(), 0);
    labels_.resize(X.rows(), 0);

    for (std::size_t iteration = 0; iteration < max_iterations_; ++iteration){
        bool changed = false;
        for (std::size_t i = 0; i < X.rows(); ++i){
            new_labels[i] = nearest_centroid(X, i);
            if (new_labels[i] != labels_[i]) {
                labels_[i] = new_labels[i];
                changed = true;
            }
        }
        if (!changed) {
            break;
        }
        Matrix sums(n_clusters_, X.cols());
        std::vector<std::size_t> counts(n_clusters_, 0);
        for (std::size_t i = 0; i < X.rows(); ++i) {
            std::size_t cluster = labels_[i];
            for (std::size_t j = 0; j < X.cols(); ++j) {
                sums(cluster, j) += X(i, j);
            }
        }
        for (std::size_t i = 0; i < n_clusters_; ++i) {
            counts[i] = std::count(labels_.begin(), labels_.end(), i);
            if (counts[i] > 0) {
                for (std::size_t j = 0; j < X.cols(); ++j) {
                    centroids_(i, j) = sums(i, j) / counts[i];
                }
            }
        }
    }

}

// TODO(toi): pour chaque ligne de X, appelle nearest_centroid et
// empile le resultat -- tres proche de KNNClassifier::predict().
std::vector<std::size_t> KMeans::predict(const Matrix& X) const {
    std::vector<std::size_t> predictions(X.rows());
    for (std::size_t i = 0; i < X.rows(); ++i) {
        predictions[i] = nearest_centroid(X, i);
    }
    return predictions;
}

const std::vector<std::size_t>& KMeans::labels() const {
    return labels_;
}

const Matrix& KMeans::centroids() const {
    return centroids_;
}

} // namespace chiikaml
