#include "chiikaml/knn.hpp"

#include <stdexcept>
#include <unordered_map>
#include <algorithm>

namespace chiikaml {

// Remarque : Matrix n'a pas de constructeur par defaut (il faut
// toujours donner rows/cols), donc X_train_ doit etre initialise
// explicitement ici, meme si fit() n'a pas encore ete appele.
// (0, 0) = une matrice vide, en attendant fit().
KNNClassifier::KNNClassifier(std::size_t k)
    : k_(k), X_train_(0, 0) {
}

// TODO(toi):
// - verifie que X.rows() == y.size() (sinon throw std::invalid_argument)
// - stocke X dans X_train_ et y (deplace ou copie) dans y_train_
void KNNClassifier::fit(const Matrix& X, std::vector<int> y) {
    
    if (X.rows() != y.size()) {
        throw std::invalid_argument("X.rows() must be equal to y.size()");
    }
    X_train_ = X;
    y_train_ = std::move(y);
}

// TODO(toi): pour chaque ligne de X, appelle predict_one et empile
// le resultat dans un vector<int> a retourner.
std::vector<int> KNNClassifier::predict(const Matrix& X) const {
    std::vector<int> predictions;
    predictions.reserve(X.rows());
    for (std::size_t i=0; i<X.rows(); ++i) {
        predictions.push_back(predict_one(X,i));
    }
    return predictions;
}

// TODO(toi): somme des carres des differences entre
// query(query_row, f) et X_train_(train_row, f) pour chaque feature f.
double KNNClassifier::squared_distance(const Matrix& query, std::size_t query_row,
                                        std::size_t train_row) const {
    double squared_distance = 0;
    for (std::size_t f=0; f<X_train_.cols(); ++f){
        double diff = query(query_row, f) - X_train_(train_row, f);
        squared_distance += diff * diff;
    }
    return squared_distance;
}

// TODO(toi):
// - calcule la distance entre `query_row` et chaque ligne de X_train_
// - trouve les k plus proches (regarde std::partial_sort ou std::nth_element)
// - vote a la majorite parmi leurs labels (regarde std::unordered_map<int, int>)
int KNNClassifier::predict_one(const Matrix& query, std::size_t query_row) const {
    std::vector<std::pair<size_t, double>> distances;
    for(std::size_t i=0; i<X_train_.rows(); ++i){
        double dist = squared_distance(query, query_row, i);
        distances.push_back({i, dist});
    }
    std::nth_element(distances.begin(), distances.begin() + k_, distances.end(),
    [](const std::pair<std::size_t, double>& a, const std::pair<std::size_t, double>& b){
        return a.second<b.second;
    });

    std::unordered_map<int, int> label_count;
    for (std::size_t i=0; i<k_; ++i){
        int label = y_train_[distances[i].first];
        label_count[label]++;
    }

    int majority_label = -1;
    int max_count = 0;

    for (const auto& pair : label_count){
        if (pair.second>max_count){
            max_count = pair.second;
            majority_label = pair.first;
        }
    }

    return majority_label; ;
}

} // namespace chiikaml
