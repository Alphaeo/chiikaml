#include "chiikaml/random_forest.hpp"

#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace chiikaml {

// TODO(toi): stocke les 4 parametres. Rien d'autre a faire ici --
// trees_ reste un vector<DecisionTreeClassifier> vide jusqu'a fit().
RandomForestClassifier::RandomForestClassifier(std::size_t n_trees, std::size_t max_depth,
                                                 std::size_t min_samples_split, unsigned int seed) {
    n_trees_ = n_trees;
    max_depth_ = max_depth;
    min_samples_split_ = min_samples_split;
    seed_ = seed;
}

// TODO(toi):
// - construis X_sample (Matrix(X.rows(), X.cols())) et y_sample
//   (vector<int> vide, avec reserve(X.rows()) si tu veux)
// - std::uniform_int_distribution<std::size_t> dist(0, X.rows() - 1);
// - repete X.rows() fois : tire un indice avec dist(rng), copie la
//   ligne correspondante de X dans la prochaine ligne de X_sample
//   (boucle sur les colonnes, comme dans DecisionTreeClassifier::build),
//   et push_back le label correspondant dans y_sample
// - renvoie {X_sample, y_sample} (une paire, construite directement,
//   pas besoin de std::make_pair)
std::pair<Matrix, std::vector<int>> RandomForestClassifier::bootstrap_sample(const Matrix& X,
                                                                               const std::vector<int>& y,
                                                                               std::mt19937& rng) {
    Matrix X_sample(X.rows(), X.cols());
    std::vector<int> y_sample;
    y_sample.reserve(X.rows());

    std::uniform_int_distribution<std::size_t> dist(0, X.rows() - 1);
    for (std::size_t i = 0; i < X.rows(); ++i) {
        std::size_t idx = dist(rng);
        for (std::size_t j = 0; j < X.cols(); ++j) {
            X_sample(i, j) = X(idx, j);
        }
        y_sample.push_back(y[idx]);
    }
    return {X_sample, y_sample};
}

// TODO(toi): identique au vote majoritaire deja ecrit dans
// KNNClassifier/DecisionTreeClassifier (unordered_map<int, size_t>
// qui compte, on garde le plus grand compte).
int RandomForestClassifier::majority_vote(const std::vector<int>& votes) {
    std::unordered_map<int, std::size_t> vote_counts;
    for (const auto& vote : votes) {
        vote_counts[vote]++;
    }

    int majority_class = votes[0];
    std::size_t max_count = 0;
    for (const auto& [label, count] : vote_counts) {
        if (count > max_count) {
            max_count = count;
            majority_class = label;
        }
    }
    return majority_class;
}

// TODO(toi), etape par etape :
//
// 1) Pre-construis les arbres, SEQUENTIELLEMENT (avant de lancer le
//    moindre thread) : trees_.reserve(n_trees_), puis n_trees_ fois
//    trees_.emplace_back(max_depth_, min_samples_split_) --
//    emplace_back construit l'objet directement dans le vector, ce
//    qui evite le probleme de copie explique dans le header.
//
// 2) Lance un thread par arbre : un std::vector<std::thread> threads,
//    et pour chaque indice t de 0 a n_trees_, threads.emplace_back
//    avec une lambda qui :
//      - capture [this, &X, &y, t] (this pour acceder a trees_/seed_,
//        X/y par reference car on ne fait que les lire, t par valeur
//        car chaque thread a besoin de SA propre copie de l'indice)
//      - construit un std::mt19937 local, seed_ + t comme graine
//      - appelle bootstrap_sample(X, y, ce generateur local)
//      - appelle trees_[t].fit(...) avec le resultat
//
// 3) Attend que tous les threads terminent : pour chaque thread dans
//    threads, appelle .join() (sinon le programme plante en quittant
//    fit() avec des threads encore actifs)
void RandomForestClassifier::fit(const Matrix& X, std::vector<int> y) {
    trees_.reserve(n_trees_);
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < n_trees_; ++t){
        trees_.emplace_back(max_depth_, min_samples_split_);
        threads.emplace_back([this, &X, &y, t]() {
            std::mt19937 rng(seed_ + t);
            auto [X_sample, y_sample] = bootstrap_sample(X, y, rng);
            trees_[t].fit(X_sample, std::move(y_sample));
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

// TODO(toi):
// - pour chaque arbre de trees_, appelle tree.predict(X) (un appel
//   PAR ARBRE, pas par point -- on reutilise le predict() deja
//   ecrit dans DecisionTreeClassifier, qui traite tout X d'un coup)
//   et stocke le resultat (un vector<int> par arbre)
// - pour chaque ligne i de X, rassemble le vote de chaque arbre a
//   la position i, et appelle majority_vote dessus
std::vector<int> RandomForestClassifier::predict(const Matrix& X) const {
    std::vector<std::vector<int>> predictions;
    for (const auto& tree : trees_) {
        auto tree_predictions = tree.predict(X);
        predictions.push_back(std::move(tree_predictions));
    }
    std::vector<int> final_predictions;
    for (std::size_t i = 0; i < X.rows(); ++i) {
        std::vector<int> votes;
        for (const auto& preds : predictions) {
            votes.push_back(preds[i]);
        }
        final_predictions.push_back(majority_vote(votes));
    }
    return final_predictions;
}

} // namespace chiikaml
