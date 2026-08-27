#pragma once

#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "chiikaml/decision_tree.hpp"
#include "chiikaml/matrix.hpp"

namespace chiikaml {

// Random Forest : un ensemble de n_trees DecisionTreeClassifier
// entraines independamment, dont on combine les predictions par
// vote majoritaire. Deux ingredients par rapport a un arbre seul :
//
//  - Bagging (Bootstrap AGGregatING) : chaque arbre n'est pas
//    entraine sur X/y tel quel, mais sur un "echantillon bootstrap"
//    -- X.rows() points tires AVEC REMISE (contrairement a
//    KMeans::fit, qui tirait des centres SANS remise via
//    std::sample). Avec remise : un meme point peut etre tire
//    plusieurs fois, d'autres pas du tout. Ca decorrele les arbres
//    entre eux (chacun voit une vue legerement differente des
//    donnees), ce qui rend le vote majoritaire plus robuste qu'un
//    arbre unique face au bruit/aux valeurs aberrantes.
//
//  - Multithreading : les n_trees arbres sont totalement
//    independants les uns des autres une fois leur echantillon
//    bootstrap tire -- aucun n'a besoin de connaitre le resultat
//    d'un autre pour s'entrainer. C'est ce qu'on appelle
//    "embarrassingly parallel" : on peut les entrainer en parallele,
//    un thread par arbre, sans aucune synchronisation complexe.
//
// Pourquoi c'est thread-safe ICI precisement (a comprendre avant
// d'ecrire le code, pas juste a appliquer) :
//  1. Chaque thread ecrit dans un index DIFFERENT de trees_ (le
//     thread t ne touche jamais que trees_[t]) -- pas d'ecriture
//     concurrente sur la meme donnee.
//  2. Chaque thread utilise son PROPRE generateur aleatoire local
//     (un std::mt19937 construit dans le thread, pas partage) --
//     std::mt19937 n'est PAS thread-safe : plusieurs threads qui
//     appelleraient le meme generateur en meme temps corrompraient
//     son etat interne (data race, comportement indefini). D'ou
//     seed_ (juste un entier, trivialement copiable) stocke comme
//     membre plutot qu'un unique std::mt19937 partage comme dans
//     KMeans -- chaque thread construit son propre generateur a
//     partir de seed_ + son numero d'arbre, pour rester reproductible
//     tout en etant independant.
//  3. X et y sont seulement LUS (jamais modifies) par les threads --
//     des lectures concurrentes sans aucune ecriture concurrente sont
//     toujours sures, pas besoin de mutex.
class RandomForestClassifier {
public:
    explicit RandomForestClassifier(std::size_t n_trees = 10, std::size_t max_depth = 5,
                                     std::size_t min_samples_split = 2, unsigned int seed = 42);

    void fit(const Matrix& X, std::vector<int> y);

    std::vector<int> predict(const Matrix& X) const;

private:
    std::size_t n_trees_;
    std::size_t max_depth_;
    std::size_t min_samples_split_;
    unsigned int seed_;

    // DecisionTreeClassifier contient un std::unique_ptr<Node> --
    // donc DecisionTreeClassifier lui-meme n'est PAS copiable (le
    // compilateur supprime automatiquement le constructeur de copie
    // d'une classe qui a un membre non copiable), seulement
    // deplacable. On ne peut donc pas faire
    // "std::vector<DecisionTreeClassifier> trees_(n_trees_, un_arbre)"
    // (ca copierait un_arbre n_trees_ fois) -- il faut construire
    // chaque arbre directement a sa place dans le vector (regarde
    // emplace_back).
    std::vector<DecisionTreeClassifier> trees_;

    // Tire X.rows() indices AVEC remise dans [0, X.rows()) en
    // utilisant rng, et renvoie le sous-ensemble (X, y)
    // correspondant. rng est passe par reference (pas par valeur) :
    // on veut que les tirages consomment/avancent l'etat du meme
    // generateur a chaque appel, pas repartir d'un generateur figé a
    // chaque fois.
    static std::pair<Matrix, std::vector<int>> bootstrap_sample(const Matrix& X, const std::vector<int>& y,
                                                                  std::mt19937& rng);

    // Vote majoritaire parmi les predictions des n_trees_ arbres pour
    // UN point. Meme principe que KNNClassifier::predict_one ou
    // DecisionTreeClassifier::majority_class -- volontairement
    // reimplemente ici plutot que partage entre les trois classes,
    // une piste de refactoring a garder en tete pour plus tard.
    static int majority_vote(const std::vector<int>& votes);
};

} // namespace chiikaml
