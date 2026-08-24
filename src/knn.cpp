#include "chiikaml/knn.hpp"

#include <stdexcept>

namespace chiikaml {

// Remarque : Matrix n'a pas de constructeur par defaut (il faut
// toujours donner rows/cols), donc X_train_ doit etre initialise
// explicitement ici, meme si fit() n'a pas encore ete appele.
// (0, 0) = une matrice vide, en attendant fit().
KNNClassifier::KNNClassifier(std::size_t k)
    : k_(k), X_train_(0, 0) {
    throw std::logic_error("KNNClassifier::KNNClassifier pas encore implemente");
}

// TODO(toi):
// - verifie que X.rows() == y.size() (sinon throw std::invalid_argument)
// - stocke X dans X_train_ et y (deplace ou copie) dans y_train_
void KNNClassifier::fit(const Matrix& X, std::vector<int> y) {
    throw std::logic_error("KNNClassifier::fit pas encore implemente");
}

// TODO(toi): pour chaque ligne de X, appelle predict_one et empile
// le resultat dans un vector<int> a retourner.
std::vector<int> KNNClassifier::predict(const Matrix& X) const {
    throw std::logic_error("KNNClassifier::predict pas encore implemente");
}

// TODO(toi): somme des carres des differences entre
// query(query_row, f) et X_train_(train_row, f) pour chaque feature f.
double KNNClassifier::squared_distance(const Matrix& query, std::size_t query_row,
                                        std::size_t train_row) const {
    throw std::logic_error("KNNClassifier::squared_distance pas encore implemente");
}

// TODO(toi):
// - calcule la distance entre `query_row` et chaque ligne de X_train_
// - trouve les k plus proches (regarde std::partial_sort ou std::nth_element)
// - vote a la majorite parmi leurs labels (regarde std::unordered_map<int, int>)
int KNNClassifier::predict_one(const Matrix& query, std::size_t query_row) const {
    throw std::logic_error("KNNClassifier::predict_one pas encore implemente");
}

} // namespace chiikaml
