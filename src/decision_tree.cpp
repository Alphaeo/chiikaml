#include "chiikaml/decision_tree.hpp"

#include <stdexcept>
#include <unordered_map>

namespace chiikaml {

// TODO(toi): stocke max_depth et min_samples_split. Rien d'autre a
// faire ici (root_ est un unique_ptr, il se construit tout seul a
// nullptr par defaut -- pas besoin de l'initialiser explicitement,
// contrairement a Matrix qui n'a pas de constructeur par defaut).
DecisionTreeClassifier::DecisionTreeClassifier(std::size_t max_depth, std::size_t min_samples_split) {
    throw std::logic_error("DecisionTreeClassifier::DecisionTreeClassifier pas encore implemente");
}

// TODO(toi):
// - compte les occurrences de chaque classe dans y (regarde
//   std::unordered_map<int, std::size_t>, deja inclus)
// - pour chaque classe c rencontree, calcule p_c = compte_c / y.size()
// - renvoie 1.0 - somme(p_c au carre)
// - cas particulier : si y est vide, renvoie 0.0 (rien a melanger)
double DecisionTreeClassifier::gini(const std::vector<int>& y) {
    throw std::logic_error("DecisionTreeClassifier::gini pas encore implemente");
}

// TODO(toi): identique au vote majoritaire de KNNClassifier::predict_one
// (un unordered_map<int, std::size_t> qui compte, puis on garde la
// classe avec le plus grand compte).
int DecisionTreeClassifier::majority_class(const std::vector<int>& y) {
    throw std::logic_error("DecisionTreeClassifier::majority_class pas encore implemente");
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
    throw std::logic_error("DecisionTreeClassifier::find_best_split pas encore implemente");
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
    throw std::logic_error("DecisionTreeClassifier::build pas encore implemente");
}

// TODO(toi):
// - stocke X et y (deplace y, comme KNNClassifier::fit)
// - root_ = build(X, y, 0)
void DecisionTreeClassifier::fit(const Matrix& X, std::vector<int> y) {
    throw std::logic_error("DecisionTreeClassifier::fit pas encore implemente");
}

// TODO(toi):
// - descend dans l'arbre a partir de root_.get() : tant que le
//   noeud courant n'est pas une feuille, regarde
//   X(row, noeud->split_feature) <= noeud->split_threshold pour
//   savoir si on continue a gauche ou a droite
// - une fois sur une feuille, renvoie son predicted_class
int DecisionTreeClassifier::predict_one(const Matrix& X, std::size_t row) const {
    throw std::logic_error("DecisionTreeClassifier::predict_one pas encore implemente");
}

// TODO(toi): pour chaque ligne de X, appelle predict_one et empile
// le resultat -- identique a KNNClassifier::predict().
std::vector<int> DecisionTreeClassifier::predict(const Matrix& X) const {
    throw std::logic_error("DecisionTreeClassifier::predict pas encore implemente");
}

} // namespace chiikaml
