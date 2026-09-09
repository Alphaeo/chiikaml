#include "chiikaml/xgboost_regressor.hpp"

#include <numeric>
#include <stdexcept>

namespace chiikaml {

// TODO(toi): stocke les six parametres dans les membres du meme nom
// (n_estimators_, learning_rate_, max_depth_, lambda_, gamma_,
// min_child_weight_). base_prediction_ et trees_ n'ont rien besoin de
// plus ici -- ils sont remplis par fit().
XGBoostRegressor::XGBoostRegressor(std::size_t n_estimators, double learning_rate, std::size_t max_depth,
                                    double lambda, double gamma, double min_child_weight) {
    throw std::logic_error("XGBoostRegressor::XGBoostRegressor pas encore implemente");
}

// TODO(toi), etape par etape :
//
// - base_prediction_ = moyenne de y (std::accumulate(y.begin(),
//   y.end(), 0.0) / y.size()).
//
// - predictions = std::vector<double>(X.rows(), base_prediction_) --
//   la prediction COURANTE de chaque exemple, mise a jour a chaque
//   round (pas juste a la fin : le round suivant a besoin des
//   gradients calcules par rapport a CES predictions, pas aux
//   originales).
//
// - pour round de 0 a n_estimators_ - 1 :
//     - calcule gradients[i] = predictions[i] - y[i] et
//       hessians[i] = 1.0 pour chaque i (derivees premiere/seconde de
//       0.5*(y_i - predictions[i])^2 par rapport a predictions[i]).
//     - construis XGBoostTree tree(lambda_, gamma_, max_depth_,
//       min_child_weight_), appelle tree.fit(X, gradients, hessians).
//     - tree_predictions = tree.predict(X).
//     - pour chaque i : predictions[i] += learning_rate_ *
//       tree_predictions[i] (met a jour AVANT le round suivant).
//     - trees_.push_back(std::move(tree)) (garde l'arbre pour
//       predict() plus tard -- rappelle-toi, XGBoostTree n'est pas
//       copiable, il faut le deplacer).
void XGBoostRegressor::fit(const Matrix& X, const std::vector<double>& y) {
    throw std::logic_error("XGBoostRegressor::fit pas encore implemente");
}

// TODO(toi), etape par etape :
//
// - preds = std::vector<double>(X.rows(), base_prediction_).
//
// - pour chaque tree de trees_ : tree_predictions = tree.predict(X) ;
//   pour chaque i : preds[i] += learning_rate_ * tree_predictions[i]
//   (exactement la meme mise a jour que dans fit(), mais en repartant
//   de base_prediction_ et en rejouant TOUS les arbres appris).
//
// - renvoie preds.
std::vector<double> XGBoostRegressor::predict(const Matrix& X) const {
    throw std::logic_error("XGBoostRegressor::predict pas encore implemente");
}

} // namespace chiikaml
