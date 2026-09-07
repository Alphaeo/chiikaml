#pragma once

#include "chiikaml/tensor.hpp"

#include <memory>
#include <unordered_set>
#include <vector>

namespace chiikaml {

class Variable;

// Represente UNE operation qui a produit une Variable : le "comment
// remonter le gradient a travers cette operation precise". Chaque
// Node connait ses ENTREES (les Variable qui ont servi a produire sa
// sortie) et sait, etant donne le gradient de la SORTIE, calculer le
// gradient de CHAQUE entree -- la regle de la chaine appliquee a une
// seule operation. Une classe concrete par operation (AddNode pour
// +, plus tard MulNode/MatmulNode/TransposeNode...).
class Node {
public:
    virtual ~Node() = default;

    // Les Variable qui ont produit la sortie de ce Node. Utilise par
    // Variable::backward() pour savoir vers OU router les gradients
    // calcules, et pour construire l'ordre topologique du graphe.
    std::vector<Variable> inputs;

    // Etant donne grad_output (le gradient de la SORTIE de
    // l'operation que ce Node represente), renvoie un gradient par
    // entree de `inputs`, DANS LE MEME ORDRE. Formule specifique a
    // chaque operation (ex: pour a+b, le gradient de la sortie se
    // propage TEL QUEL vers a ET vers b, puisque d(a+b)/da = d(a+b)/db = 1).
    virtual std::vector<Tensor> backward(const Tensor& grad_output) const = 0;
};

namespace detail {

// Etat partage d'une Variable -- separe de Variable elle-meme pour
// qu'une meme Variable LOGIQUE, utilisee plusieurs fois dans le
// graphe (ex : c = a + a), partage UN SEUL gradient accumule entre
// toutes ses copies. Exactement le meme principe que data_ dans
// Tensor (un shared_ptr pour que les vues partagent le meme buffer),
// applique ici a TOUT l'etat d'autograd plutot qu'a un simple buffer
// numerique.
struct VariableImpl {
    Tensor data;
    Tensor grad;
    bool requires_grad;
    std::shared_ptr<Node> grad_fn; // nullptr pour une Variable feuille
};

} // namespace detail

// Une valeur "differentiable" : une Tensor (la valeur, comme
// d'habitude) plus tout ce qu'il faut pour remonter un gradient a
// travers elle. Copier une Variable est bon marche (copie d'un
// shared_ptr) et TOUTES les copies d'une meme Variable partagent le
// meme gradient accumule -- voir detail::VariableImpl ci-dessus pour
// le pourquoi.
class Variable {
public:
    // Construit une Variable. Si grad_fn est nullptr (valeur par
    // defaut), c'est une Variable "feuille" -- typiquement une entree
    // du reseau ou un parametre appris, rien a remonter au-dela.
    // Sinon, c'est le RESULTAT d'une operation (utilise en interne
    // par operator+ etc., pas destine a etre appele directement avec
    // un grad_fn non-null depuis l'exterieur de la classe).
    explicit Variable(Tensor data, bool requires_grad = false,
                       std::shared_ptr<Node> grad_fn = nullptr);

    const Tensor& data() const { return impl_->data; }
    const Tensor& grad() const { return impl_->grad; }
    bool requires_grad() const { return impl_->requires_grad; }

    // Remet le gradient accumule a zero (meme forme que data(),
    // toutes les valeurs a 0.0) -- a appeler entre deux passes
    // d'entrainement (sinon les gradients de passes successives
    // s'additionneraient entre elles, pas seulement au sein d'une
    // meme passe backward()).
    void zero_grad();

    // Ajoute delta au gradient accumule de cette Variable (au lieu
    // de l'ecraser) SEULEMENT si requires_grad() est vrai (sinon ne
    // fait rien -- pas la peine d'accumuler un gradient dont personne
    // ne se servira). C'est cette accumulation (plutot qu'un simple
    // remplacement) qui rend le "fan-out" correct : quand une meme
    // Variable est utilisee plusieurs fois dans le graphe, chaque
    // utilisation contribue sa part au gradient total -- la regle de
    // la chaine multivariee est une SOMME sur tous les chemins.
    void accumulate_grad(const Tensor& delta);

    // Lance la retropropagation a partir de CETTE Variable (le
    // "sommet" du graphe, typiquement une perte). Parcourt le graphe
    // en ordre topologique inverse, accumule le gradient de chaque
    // Variable rencontree en cours de route.
    //
    // Limitation assumee de cette v1 : data() doit avoir exactement 1
    // element (backward() sur un scalaire, le cas normal pour une
    // perte). Leve std::invalid_argument sinon.
    void backward();

    // Addition element par element (comme Tensor::operator+, qu'elle
    // utilise en interne pour la partie "valeur"), mais construit en
    // plus un AddNode qui saura repropager le gradient vers les deux
    // entrees lors d'un backward() ulterieur.
    Variable operator+(const Variable& other) const;

private:
    std::shared_ptr<detail::VariableImpl> impl_;

    // Construit l'ordre topologique du graphe atteignable depuis
    // cette Variable (DFS en postorder : chaque Variable n'est
    // ajoutee a `topo` qu'APRES avoir recurse dans toutes ses
    // entrees) -- utilise par backward(). `visited` identifie une
    // Variable deja rencontree par l'adresse de son VariableImpl
    // partage (impl_.get()) : indispensable des qu'une meme Variable
    // logique est atteinte par plusieurs chemins (fan-out), sinon on
    // la visiterait -- et l'ajouterait a topo -- plusieurs fois.
    void build_topo(std::vector<Variable>& topo, std::unordered_set<const void*>& visited) const;
};

} // namespace chiikaml
