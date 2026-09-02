#include "chiikaml/decision_tree.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace chiikaml {

// TODO(toi): stocke max_depth et min_samples_split. Rien d'autre a
// faire ici (root_ est un unique_ptr, il se construit tout seul a
// nullptr par defaut -- pas besoin de l'initialiser explicitement,
// contrairement a Matrix qui n'a pas de constructeur par defaut).
DecisionTreeClassifier::DecisionTreeClassifier(std::size_t max_depth, std::size_t min_samples_split) {
    max_depth_ = max_depth;
    min_samples_split_ = min_samples_split;
}

// TODO(toi):
// - compte les occurrences de chaque classe dans y (regarde
//   std::unordered_map<int, std::size_t>, deja inclus)
// - pour chaque classe c rencontree, calcule p_c = compte_c / y.size()
// - renvoie 1.0 - somme(p_c au carre)
// - cas particulier : si y est vide, renvoie 0.0 (rien a melanger)
double DecisionTreeClassifier::gini(const std::vector<int>& y) {
    if (y.empty()) {
        return 0;
    }
    std::unordered_map<int , std::size_t> class_counts;
    for (const auto& label : y) {
        class_counts[label]++;
    }

    double gini = 1.0;
    double total = y.size();
    for (const auto& [label, count] : class_counts) {
        double p_c = static_cast<double>(count) / total;
        gini -= p_c * p_c;
    }
    return gini;
}

// Meme formule que gini(), mais sans jamais parcourir y : les
// comptes sont deja donnes (maintenus par find_best_split() au fur
// et a mesure). O(nombre de classes distinctes), pas O(n).
double DecisionTreeClassifier::gini_from_counts(const std::unordered_map<int, std::size_t>& counts,
                                                 std::size_t total) {
    if (total == 0) {
        return 0.0;
    }
    double g = 1.0;
    for (const auto& [label, count] : counts) {
        double p = static_cast<double>(count) / static_cast<double>(total);
        g -= p * p;
    }
    return g;
}

// TODO(toi): identique au vote majoritaire de KNNClassifier::predict_one
// (un unordered_map<int, std::size_t> qui compte, puis on garde la
// classe avec le plus grand compte).
int DecisionTreeClassifier::majority_class(const std::vector<int>& y) {
    std::unordered_map<int, std::size_t> class_counts;
    for (const auto& label : y) {
        class_counts[label]++;
    }

    int majority_class = y[0];
    std::size_t max_count = 0;
    for (const auto& [label, count] : class_counts) {
        if (count > max_count) {
            max_count = count;
            majority_class = label;
        }
    }
    return majority_class;
}

// TODO(toi), etape par etape -- la fonction la plus dense du module :
//
// - garde une trace du meilleur score trouve jusqu'ici (ex: un
//   double best_score, initialise a une valeur "pire que tout",
//   comme 1.0 -- le gini max possible -- ou plus grand) et un bool
//   "found" pour savoir si on a trouve au moins un split valide
//
// - pour chaque feature f de 0 a X.cols() :
//     - pour chaque ligne i de X, la valeur candidate de seuil peut
//       etre X(i, f) elle-meme (approche simple : tester la valeur
//       de chaque point comme seuil, pas seulement les "milieux
//       entre valeurs triees" mentionnes dans le header -- suffisant
//       et plus simple pour une premiere version)
//     - pour ce seuil candidat, partitionne (mentalement, sans
//       modifier X/y) les labels de y en y_left (points ou
//       X(i, f) <= seuil) et y_right (les autres)
//     - si y_left ou y_right est vide, ce candidat ne separe rien,
//       ignore-le
//     - calcule le score pondere :
//       (y_left.size() * gini(y_left) + y_right.size() * gini(y_right)) / y.size()
//     - si ce score est meilleur (plus petit) que best_score,
//       memorise-le avec la feature et le seuil courants, et passe
//       found a true
//
// - a la fin, ecrit best_feature/best_threshold dans les parametres
//   de sortie (seulement si found est true) et renvoie found
bool DecisionTreeClassifier::find_best_split(const Matrix& X, const std::vector<int>& y,
                                              std::size_t& best_feature, double& best_threshold) {
    double best_score = 1.0; // pire score possible
    bool found = false;
    const std::size_t n = y.size();

    for (std::size_t f = 0; f < X.cols(); ++f) {
        // Indices des points, tries par valeur croissante sur cette
        // feature -- std::iota remplit order = {0, 1, ..., n-1},
        // std::sort les reordonne ensuite selon X(idx, f).
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&X, f](std::size_t a, std::size_t b) { return X(a, f) < X(b, f); });

        // Etat initial du balayage : tout le monde est "a droite",
        // rien "a gauche".
        std::unordered_map<int, std::size_t> left_counts;
        std::unordered_map<int, std::size_t> right_counts;
        for (std::size_t idx : order) {
            right_counts[y[idx]]++;
        }
        std::size_t left_size = 0;
        std::size_t right_size = n;

        // Deplace les points un par un de droite vers gauche, dans
        // l'ordre trie -- a chaque etape, "gauche" = tous les points
        // dont la valeur est <= celle du point qu'on vient de
        // deplacer.
        for (std::size_t pos = 0; pos + 1 < n; ++pos) {
            std::size_t idx = order[pos];
            int label = y[idx];

            left_counts[label]++;
            --right_counts[label];
            if (right_counts[label] == 0) {
                right_counts.erase(label);
            }
            ++left_size;
            --right_size;

            double current_value = X(idx, f);
            double next_value = X(order[pos + 1], f);
            if (current_value == next_value) {
                // Un autre point partage la meme valeur : ne pas
                // couper "au milieu" de valeurs egales, attendre de
                // les avoir toutes deplacees a gauche.
                continue;
            }

            double score = (static_cast<double>(left_size) * gini_from_counts(left_counts, left_size) +
                             static_cast<double>(right_size) * gini_from_counts(right_counts, right_size)) /
                            static_cast<double>(n);

            if (score < best_score) {
                best_score = score;
                best_feature = f;
                best_threshold = current_value;
                found = true;
            }
        }
    }
    return found;
}

// TODO(toi), etape par etape :
//
// - cas d'arret -> cree une feuille (Node avec is_leaf = true,
//   predicted_class = majority_class(y)) si au moins une de ces
//   conditions est vraie :
//     - gini(y) == 0.0 (deja pur)
//     - depth >= max_depth_
//     - y.size() < min_samples_split_
//
// - sinon, appelle find_best_split(X, y, best_feature, best_threshold).
//   Si elle renvoie false (aucun split valide trouve), cree aussi
//   une feuille (meme logique que ci-dessus).
//
// - sinon, construis les deux sous-ensembles : parcours les lignes
//   de X, et pour chacune, regarde si X(ligne, best_feature) <=
//   best_threshold pour savoir si elle va dans le groupe gauche ou
//   droit -- construis X_left/y_left et X_right/y_right (deux
//   Matrix + deux vector<int>, remplis au fur et a mesure ; pour
//   Matrix, il n'y a pas de push_back, donc soit tu comptes d'abord
//   la taille de chaque groupe pour construire les deux Matrix a la
//   bonne taille direction, soit tu accumules d'abord dans des
//   vector<vector<double>> puis tu recopies -- a toi de choisir une
//   approche)
//
// - cree le noeud (is_leaf = false, split_feature, split_threshold),
//   node->left = build(X_left, y_left, depth + 1),
//   node->right = build(X_right, y_right, depth + 1), renvoie node
std::unique_ptr<DecisionTreeClassifier::Node> DecisionTreeClassifier::build(const Matrix& X,
                                                                             const std::vector<int>& y,
                                                                             std::size_t depth) const {
    if (gini(y) == 0.0 || depth >= max_depth_ || y.size() < min_samples_split_) {
        auto leaf_node = std::make_unique<Node>();
        leaf_node->is_leaf = true;
        leaf_node->predicted_class = majority_class(y);
        return leaf_node;
    }

    std::size_t best_feature;
    double best_feature_threshold;
    if (!find_best_split(X, y, best_feature, best_feature_threshold)) {
        auto leaf_node = std::make_unique<Node>();
        leaf_node->is_leaf = true;
        leaf_node->predicted_class = majority_class(y);
        return leaf_node;
    }

    std::vector<int> y_left;
    std::vector<int> y_right;
    for (std::size_t i = 0; i < X.rows(); ++i) {
        if (X(i, best_feature) <= best_feature_threshold) {
            y_left.push_back(y[i]);
        } else {
            y_right.push_back(y[i]);
        }
    }
    Matrix X_left(y_left.size(), X.cols());
    Matrix X_right(y_right.size(), X.cols());
    std::size_t left_index = 0;
    std::size_t right_index = 0;

    for (std::size_t i = 0; i < X.rows(); ++i) {
        if (X(i, best_feature) <= best_feature_threshold) {
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

    auto node = std::make_unique<Node>();
    node->is_leaf = false;
    node->split_feature = best_feature;
    node->split_threshold = best_feature_threshold;
    node->left = build(X_left, y_left, depth + 1);
    node->right = build(X_right, y_right, depth + 1);
    return node;
    


}

void DecisionTreeClassifier::fit(const Matrix& X, std::vector<int> y) {
    root_ = build(X, y, 0);
}

// TODO(toi):
// - descend dans l'arbre a partir de root_.get() : tant que le
//   noeud courant n'est pas une feuille, regarde
//   X(row, noeud->split_feature) <= noeud->split_threshold pour
//   savoir si on continue a gauche ou a droite
// - une fois sur une feuille, renvoie son predicted_class
int DecisionTreeClassifier::predict_one(const Matrix& X, std::size_t row) const {
    auto node = root_.get();
    while (node && !node->is_leaf) {
        if (X(row, node->split_feature) <= node->split_threshold) {
            node = node->left.get();
        } else {
            node = node->right.get();
        }
    }
    return node ? node->predicted_class : -1;
}

// TODO(toi): pour chaque ligne de X, appelle predict_one et empile
// le resultat -- identique a KNNClassifier::predict().
std::vector<int> DecisionTreeClassifier::predict(const Matrix& X) const {
    std::vector<int> predictions;
    for (std::size_t i = 0; i < X.rows(); ++i) {
        predictions.push_back(predict_one(X, i));
    }
    return predictions;
}

} // namespace chiikaml
