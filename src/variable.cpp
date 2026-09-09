#include "chiikaml/variable.hpp"

#include <stdexcept>

namespace chiikaml {

namespace {

// Node concret pour operator+ : d(a+b)/da = 1 et d(a+b)/db = 1
// (element par element), donc le gradient de la sortie se propage
// TEL QUEL vers les deux entrees -- aucun calcul a faire, juste le
// renvoyer deux fois (une fois par entree, dans l'ordre de `inputs`).
//
// TODO(toi) : implemente backward() pour renvoyer {grad_output,
// grad_output} (un vector<Tensor> a 2 elements).
class AddNode : public Node {
public:
    std::vector<Tensor> backward(const Tensor& grad_output) const override {
        return {grad_output, grad_output};
    }
};

} // namespace

// TODO(toi) :
// - construis impl_ = std::make_shared<detail::VariableImpl>(...)
//   avec : data = le Tensor recu (deplace-le, std::move), grad = un
//   Tensor de meme forme que data, deja a 0.0 (Tensor(data.shape())
//   est deja zero-initialise par son propre constructeur -- pas
//   besoin de boucle), requires_grad = le bool recu, grad_fn = le
//   shared_ptr<Node> recu (deja nullptr par defaut si pas precise a
//   l'appel).
//
// Attention a l'ordre : construis d'abord la forme/valeur de `grad`
// AVANT de deplacer `data` dans la struct si tu utilises data.shape()
// -- une fois data deplace, son etat n'est plus garanti utilisable.
Variable::Variable(Tensor data, bool requires_grad, std::shared_ptr<Node> grad_fn) {
    impl_ = std::make_shared<detail::VariableImpl>(
        std::move(data),
        Tensor(data.shape()), // grad initialement zero-initialise
        requires_grad,
        std::move(grad_fn)
    );
}

// TODO(toi) : remplace impl_->grad par un nouveau Tensor de la meme
// forme que impl_->data (a nouveau, deja zero-initialise par
// construction -- pas de boucle a ecrire).
void Variable::zero_grad() {
    impl_->grad = Tensor(impl_->data.shape());
}

// TODO(toi) : si requires_grad() est vrai, impl_->grad = impl_->grad
// + delta (reutilise Tensor::operator+, deja ecrit). Si faux, ne fais
// rien.
void Variable::accumulate_grad(const Tensor& delta) {
    if (requires_grad()) {
        impl_->grad = impl_->grad + delta;
    }
}

// TODO(toi), etape par etape :
//
// - si visited contient deja impl_.get() : return immediatement (deja
//   visite, rien a refaire -- cf commentaire dans le header sur le
//   fan-out).
// - sinon, insere impl_.get() dans visited.
// - si impl_->grad_fn n'est pas nullptr (cette Variable est le
//   resultat d'une operation, pas une feuille) : pour CHAQUE Variable
//   de impl_->grad_fn->inputs, appelle input.build_topo(topo,
//   visited) recursivement -- AVANT d'ajouter *this a topo (c'est le
//   coeur du postorder : les entrees doivent apparaitre dans topo
//   avant la sortie qui en depend).
// - enfin, topo.push_back(*this).
void Variable::build_topo(std::vector<Variable>& topo, std::unordered_set<const void*>& visited) const {
    if (visited.find(impl_.get()) != visited.end()) {
        return; // deja visite
    }
    visited.insert(impl_.get());
    if (impl_->grad_fn != nullptr) {
        for (const auto& input : impl_->grad_fn->inputs) {
            input.build_topo(topo, visited);
        }
    }
    topo.push_back(*this);
}

// TODO(toi), etape par etape :
//
// - si data().size() != 1, throw std::invalid_argument (limitation
//   v1 : backward() seulement depuis un scalaire).
//
// - construis topo (un std::vector<Variable> vide) et visited (un
//   std::unordered_set<const void*> vide), puis appelle
//   build_topo(topo, visited) sur *this pour le remplir.
//
// - seme le gradient de *this a 1.0 : construis un Tensor de la forme
//   data().shape() (deja zero-initialise), mets son unique element a
//   1.0 (un index a 0 sur chaque dimension : std::vector<std::size_t>
//   zero_index(data().ndim(), 0), puis seed(zero_index) = 1.0), et
//   appelle accumulate_grad(seed) sur *this.
//
// - parcours topo A L'ENVERS (de la fin vers le debut -- utilise des
//   reverse iterators, rbegin()/rend(), comme dans transpose()). Pour
//   chaque Variable `node` rencontree :
//     - si node n'a pas de grad_fn (feuille) : rien a propager plus
//       loin, continue.
//     - sinon : grad_inputs = node.impl_->grad_fn->backward(node.grad())
//       (un vector<Tensor>, un par entree). Pour chaque i, appelle
//       node.impl_->grad_fn->inputs[i].accumulate_grad(grad_inputs[i]).
//
// Attention a l'ordre du parcours : c'est essentiel que `node.grad()`
// soit COMPLET (toutes ses contributions accumulees, y compris celles
// venant d'un eventuel fan-out plus loin dans le graphe) avant
// d'appeler backward() dessus -- c'est exactement ce que garantit le
// parcours a l'envers de l'ordre topologique.
void Variable::backward() {
    if (data().size() != 1) {
        throw std::invalid_argument("Variable::backward requires a scalar Variable (size() == 1)");
    }

    std::vector<Variable> topo;
    std::unordered_set<const void*> visited;
    build_topo(topo, visited);

    std::vector<std::size_t> zero_index(data().ndim(), 0);
    Tensor seed(data().shape());
    seed(zero_index) = 1.0;
    accumulate_grad(seed);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        const Variable& node = *it;
        if (node.impl_->grad_fn != nullptr) {
            std::vector<Tensor> grad_inputs = node.impl_->grad_fn->backward(node.grad());
            for (std::size_t i = 0; i < grad_inputs.size(); ++i) {
                node.impl_->grad_fn->inputs[i].accumulate_grad(grad_inputs[i]);
            }
        }
    }
}

// TODO(toi) :
// - result_data = data() + other.data() (Tensor::operator+, deja
//   ecrit).
// - result_requires_grad = requires_grad() || other.requires_grad().
// - si result_requires_grad : cree un std::make_shared<AddNode>(),
//   remplis son ->inputs avec {*this, other} (dans cet ordre -- c'est
//   cet ordre que AddNode::backward() doit respecter), utilise-le
//   comme grad_fn du Variable renvoye. Sinon (aucune des deux entrees
//   ne requiert de gradient), renvoie un Variable avec grad_fn =
//   nullptr (pas la peine de construire un noeud que personne ne
//   parcourra jamais).
// - renvoie Variable(result_data, result_requires_grad, le grad_fn
//   choisi).
//
// Rappel (comme pour Tensor::operator+) : tu es dans une methode de
// Variable, donc tu as acces aux membres prives de `other` (une autre
// instance de la MEME classe) -- meme regle que toujours.
Variable Variable::operator+(const Variable& other) const {
    Tensor result_data = data() + other.data();
    bool result_requires_grad = requires_grad() || other.requires_grad();
    std::shared_ptr<Node> grad_fn = nullptr;

    if (result_requires_grad) {
        auto add_node = std::make_shared<AddNode>();
        add_node->inputs = {*this, other};
        grad_fn = add_node;
    }

    return Variable(result_data, result_requires_grad, grad_fn);
}

} // namespace chiikaml
