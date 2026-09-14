#include "chiikaml/xgboost_tree.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace chiikaml {

// TODO(toi): stocke les quatre parametres dans les membres du meme
// nom (lambda_, gamma_, max_depth_, min_child_weight_). Rien d'autre
// a faire (root_ se construit tout seul a nullptr).
XGBoostTree::XGBoostTree(double lambda, double gamma, std::size_t max_depth, double min_child_weight) {
    lambda_ = lambda;
    gamma_ = gamma;
    max_depth_ = max_depth;
    min_child_weight_ = min_child_weight;
}

// TODO(toi): renvoie -G / (H + lambda) -- une seule ligne, mais
// relis le commentaire de classe pour bien comprendre CE QUE ce
// nombre represente avant de l'ecrire (ce n'est pas une formule a
// apprendre par coeur, c'est la solution exacte d'une minimisation).
double XGBoostTree::leaf_weight(double G, double H, double lambda) {
    return -G / (H + lambda);
}

// TODO(toi), etape par etape -- la fonction la plus dense du module,
// tres proche dans sa structure de DecisionTreeClassifier::find_best_split
// (relis-la si besoin) mais avec des SOMMES a la place de comptes de
// classes, et une formule de gain differente :
//
// - calcule G_total = somme(gradients), H_total = somme(hessians)
//   sur TOUT le sous-ensemble recu (pas juste un candidat).
//
// - best_gain = 0.0 (un split n'est garde QUE si son gain depasse
//   strictement 0 -- gamma est deja soustrait dans la formule, donc
//   "gain > 0" signifie deja "gain > gamma" implicitement).
//   found = false.
//
// - pour chaque feature f de 0 a X.cols() :
//     - construis order (indices 0..n-1 tries par X(idx, f)
//       croissant) -- exactement comme DecisionTreeClassifier
//       (std::iota puis std::sort avec un comparateur sur X(a, f) < X(b, f)).
//     - GL = 0.0, HL = 0.0 (rien a gauche au depart).
//     - pour pos de 0 a n-2 :
//         - idx = order[pos]
//         - GL += gradients[idx] ; HL += hessians[idx]
//         - GR = G_total - GL ; HR = H_total - HL
//         - si X(idx, f) == X(order[pos+1], f) : continue (ne pas
//           couper au milieu de valeurs egales, meme piege que
//           DecisionTreeClassifier).
//         - si HL < min_child_weight_ OU HR < min_child_weight_ :
//           continue (coupure rejetee ICI, mais le balayage continue
//           -- un seuil plus loin peut redevenir valide).
//         - gain = 0.5 * (GL*GL/(HL+lambda_) + GR*GR/(HR+lambda_)
//                          - G_total*G_total/(H_total+lambda_)) - gamma_
//         - si gain > best_gain : memorise best_feature = f,
//           best_threshold = X(idx, f), best_gain = gain, found = true.
//
// - renvoie found (best_feature/best_threshold/best_gain sont deja a
//   jour si found est vrai).
bool XGBoostTree::find_best_split(const Matrix& X, const std::vector<double>& gradients,
                                   const std::vector<double>& hessians, std::size_t& best_feature,
                                   double& best_threshold, double& best_gain) const {
    std::size_t n = X.rows();
    if (n == 0) return false;

    double G_total = std::accumulate(gradients.begin(), gradients.end(), 0.0);
    double H_total = std::accumulate(hessians.begin(), hessians.end(), 0.0);

    bool found = false;
    best_gain = 0.0;

    for (std::size_t f = 0; f < X.cols(); ++f) {
        // Step 1: Build the order
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return X(a, f) < X(b, f);
        });

        double GL = 0.0, HL = 0.0;
        for (std::size_t pos = 0; pos < n - 1; ++pos) {
            std::size_t idx = order[pos];
            GL += gradients[idx];
            HL += hessians[idx];
            double GR = G_total - GL;
            double HR = H_total - HL;

            if (X(idx, f) == X(order[pos + 1], f)) continue;
            if (HL < min_child_weight_ || HR < min_child_weight_) continue;

            double gain = 0.5 * (GL * GL / (HL + lambda_) + GR * GR / (HR + lambda_) - G_total * G_total / (H_total + lambda_)) - gamma_;
            if (gain > best_gain) {
                best_feature = f;
                best_threshold = X(idx, f);
                best_gain = gain;
                found = true;
            }
        }
    }

    return found;
}

// TODO(toi), etape par etape -- tres proche de DecisionTreeClassifier::build :
//
// - G = somme(gradients), H = somme(hessians) sur ce sous-ensemble.
//
// - cas d'arret -> cree une feuille (is_leaf = true, leaf_value =
//   leaf_weight(G, H, lambda_)) si depth >= max_depth_.
//
// - sinon, appelle find_best_split(...). Si elle renvoie false
//   (aucune coupure valide), cree aussi une feuille (meme formule).
//
// - sinon, partitionne X/gradients/hessians en deux sous-ensembles
//   selon X(i, best_feature) <= best_threshold -- meme technique que
//   DecisionTreeClassifier::build (compte d'abord la taille de
//   chaque groupe pour construire les deux Matrix a la bonne taille,
//   ou accumule dans des vector<vector<double>> puis recopie).
//
// - cree le noeud interne (is_leaf = false, split_feature,
//   split_threshold), node->left = build(X_left, gradients_left,
//   hessians_left, depth + 1), node->right = build(..., depth + 1),
//   renvoie node.
std::unique_ptr<XGBoostTree::Node> XGBoostTree::build(const Matrix& X, const std::vector<double>& gradients,
                                                        const std::vector<double>& hessians,
                                                        std::size_t depth) const {
    double G = std::accumulate(gradients.begin(), gradients.end(), 0.0);
    double H = std::accumulate(hessians.begin(), hessians.end(), 0.0);
    
    if (depth >= max_depth_) {
        auto leaf_node = std::make_unique<Node>();
        leaf_node->is_leaf = true;
        leaf_node->leaf_value = leaf_weight(G, H, lambda_);
        return leaf_node;
    }

    std::size_t best_feature;
    double best_threshold, best_gain;
    if (!find_best_split(X, gradients, hessians, best_feature, best_threshold, best_gain)) {
        auto leaf_node = std::make_unique<Node>();
        leaf_node->is_leaf = true;
        leaf_node->leaf_value = leaf_weight(G, H, lambda_);
        return leaf_node;
    }

    // Step 3: Partition the data (deux passes, comme DecisionTreeClassifier::build :
    // Matrix n'a pas de constructeur par defaut ni de push_back, il faut
    // connaitre la taille de chaque groupe AVANT de construire X_left/X_right).
    std::vector<double> gradients_left, gradients_right;
    std::vector<double> hessians_left, hessians_right;

    for (std::size_t i = 0; i < X.rows(); ++i) {
        if (X(i, best_feature) <= best_threshold) {
            gradients_left.push_back(gradients[i]);
            hessians_left.push_back(hessians[i]);
        } else {
            gradients_right.push_back(gradients[i]);
            hessians_right.push_back(hessians[i]);
        }
    }

    Matrix X_left(gradients_left.size(), X.cols());
    Matrix X_right(gradients_right.size(), X.cols());
    std::size_t left_index = 0;
    std::size_t right_index = 0;

    for (std::size_t i = 0; i < X.rows(); ++i) {
        if (X(i, best_feature) <= best_threshold) {
            for (std::size_t j = 0; j < X.cols(); ++j) {
                X_left(left_index, j) = X(i, j);
            }
            ++left_index;
        } else {
            for (std::size_t j = 0; j < X.cols(); ++j) {
                X_right(right_index, j) = X(i, j);
            }
            ++right_index;
        }
    }

    // Step 4: Create the internal node
    auto node = std::make_unique<Node>();
    node->is_leaf = false;
    node->split_feature = best_feature;
    node->split_threshold = best_threshold;

    // Step 5: Recursively build the left and right subtrees
    node->left = build(X_left, gradients_left, hessians_left, depth + 1);
    node->right = build(X_right, gradients_right, hessians_right, depth + 1);

    return node;
}

// TODO(toi): root_ = build(X, gradients, hessians, 0).
void XGBoostTree::fit(const Matrix& X, const std::vector<double>& gradients,
                       const std::vector<double>& hessians) {
    root_ = build(X, gradients, hessians, 0);
}

// TODO(toi): descend dans l'arbre a partir de root_.get(), exactement
// comme DecisionTreeClassifier::predict_one, jusqu'a une feuille ;
// renvoie son leaf_value.
double XGBoostTree::predict_one(const Matrix& X, std::size_t row) const {
    const Node* node = root_.get();
    while (node) {
        if (node->is_leaf) {
            return node->leaf_value;
        }
        if (X(row, node->split_feature) <= node->split_threshold) {
            node = node->left.get();
        } else {
            node = node->right.get();
        }
    }
    throw std::logic_error("XGBoostTree::predict_one pas encore implemente");
}

// TODO(toi): pour chaque ligne de X, appelle predict_one et empile le
// resultat -- identique a DecisionTreeClassifier::predict.
std::vector<double> XGBoostTree::predict(const Matrix& X) const {
    std::vector<double> predictions;
    for (std::size_t i = 0; i < X.rows(); ++i) {
        predictions.push_back(predict_one(X, i));
    }
    return predictions;
}

} // namespace chiikaml
