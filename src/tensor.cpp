#include "chiikaml/tensor.hpp"

#include <numeric>
#include <stdexcept>

namespace chiikaml {

// TODO(toi), etape par etape :
//
// - stocke shape (deplace-le : shape_ = std::move(shape), evite une
//   copie inutile du vector -- meme idee que std::move(y) dans
//   KNNClassifier::fit)
//
// - calcule la taille totale (le nombre d'elements) : le produit de
//   toutes les dimensions de shape_. std::accumulate peut faire ca
//   en une ligne : std::accumulate(shape_.begin(), shape_.end(),
//   std::size_t{1}, std::multiplies<std::size_t>()) -- ou une boucle
//   "a la main" qui multiplie un total en le parcourant, si tu
//   preferes rester sur ce que tu connais deja.
//
// - calcule strides_ : un vector<size_t> de la meme taille que
//   shape_. La regle : strides_[ndim-1] = 1 (la derniere dimension
//   est toujours contigue), puis en repartant de la fin vers le
//   debut, strides_[k] = strides_[k+1] * shape_[k+1] pour k de
//   ndim-2 a 0. Exemple concret pour shape_ = {2, 3, 4} :
//     strides_[2] = 1
//     strides_[1] = strides_[2] * shape_[2] = 1 * 4 = 4
//     strides_[0] = strides_[1] * shape_[1] = 4 * 3 = 12
//   (verifie que ca correspond a l'intuition : avancer d'un cran sur
//   la dimension 0 (la plus "large") saute par-dessus 3*4 = 12
//   elements, exactement les 12 elements de la "tranche" suivante)
//
// - construit data_ : un std::make_shared<std::vector<double>>(taille_totale, 0.0)
Tensor::Tensor(std::vector<std::size_t> shape) {

    shape_ = std::move(shape);
    std::size_t total_size = std::accumulate(shape_.begin(), shape_.end(), std::size_t{1}, std::multiplies<std::size_t>());
    strides_.resize(shape_.size());
    strides_[shape_.size() - 1] = 1;
    for (std::size_t k = shape_.size() - 1; k > 0; --k) {
        strides_[k - 1] = strides_[k] * shape_[k];
    }
    data_ = std::make_shared<std::vector<double>>(total_size, 0.0);

}

std::size_t Tensor::size() const {
    return data_->size();
}

// TODO(toi): produit scalaire entre `indices` et strides_ -- pour
// chaque dimension k, indices[k] * strides_[k], somme le tout.
std::size_t Tensor::flat_index(const std::vector<std::size_t>& indices) const {
    std::size_t index = 0;
    for (std::size_t k = 0; k < indices.size(); ++k) {
        index += indices[k] * strides_[k];
    }
    return index;
}

// TODO(toi): appelle flat_index(indices), utilise le resultat pour
// indexer (*data_)[...] (data_ est un shared_ptr<vector<double>> --
// (*data_) donne acces au vector qu'il possede, comme dereferencer
// un pointeur classique).
double& Tensor::operator()(const std::vector<std::size_t>& indices) {
    return (*data_)[flat_index(indices)];
}

double Tensor::operator()(const std::vector<std::size_t>& indices) const {
    return (*data_)[flat_index(indices)];
}

// TODO(toi):
// - si shape_ != other.shape_, throw std::invalid_argument (les
//   vector<size_t> se comparent directement avec == : compare taille
//   ET contenu)
// - cree Tensor result(shape_)
// - pour cette version, les deux tenseurs sont garantis contigus
//   (aucune vue/transposition n'existe encore dans ce module) --
//   donc PAS BESOIN de reconstruire des indices multi-dimensionnels :
//   une simple boucle sur size() qui additionne directement les
//   buffers plats suffit. (*data_)[i] + (*other.data_)[i] pour
//   chaque i, ecrit dans (*result.data_)[i]. Attention : result.data_
//   est prive, mais tu es DANS une methode de Tensor, donc tu as
//   acces aux membres prives d'un autre Tensor (result, other) --
//   meme regle que ce qu'on avait vu pour Matrix::operator+.
Tensor Tensor::operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensors must have the same shape");
    }
    Tensor result(shape_);
    for (std::size_t i = 0; i< size(); ++i){
        (*result.data_)[i] = (*data_)[i] + (*other.data_)[i];
    }
    return result;
}

// TODO(toi), etape par etape :
//
// - construis un vector<size_t> reversed_shape qui contient les
//   elements de shape_ dans l'ordre INVERSE. Le plus simple :
//   std::vector<std::size_t> reversed_shape(shape_.rbegin(), shape_.rend());
//   -- rbegin()/rend() sont des iterateurs "inverses" : parcourir de
//   rbegin() a rend() revient a parcourir le vector du dernier au
//   premier element. Construire un nouveau vector directement a
//   partir de ces iterateurs le remplit deja dans le bon ordre, pas
//   besoin de boucle a la main.
//
// - fais pareil pour strides_ -> reversed_strides
//
// - renvoie Tensor(reversed_shape, reversed_strides, data_) -- ca
//   appelle le constructeur PRIVE (celui a 3 arguments, deja ecrit
//   dans le header) qui ne fait AUCUNE allocation, juste une init de
//   membres. `data_` est passe tel quel (pas *data_, pas de copie du
//   contenu) : c'est un shared_ptr, le copier fait juste partager le
//   meme buffer sous-jacent avec le nouveau Tensor renvoye.
Tensor Tensor::transpose() const {
    std::vector<std::size_t> reversed_shape(shape_.rbegin(), shape_.rend());
    std::vector<std::size_t> reversed_strides(strides_.rbegin(), strides_.rend());
    return Tensor(reversed_shape, reversed_strides, data_);
}

// TODO(toi), etape par etape :
//
// - calcule le nombre total d'elements de new_shape (meme technique
//   que dans le constructeur : std::accumulate avec
//   std::multiplies<std::size_t>())
//
// - si ce total est different de size(), throw std::invalid_argument
//
// - calcule les nouveaux strides pour new_shape -- EXACTEMENT le
//   meme algorithme que dans le constructeur (dernier stride = 1,
//   puis en repartant de la fin : strides[k] = strides[k+1] * shape[k+1]).
//   Tu peux copier-coller/adapter la boucle que tu as deja ecrite
//   dans le constructeur, juste sur new_shape au lieu de shape_.
//
// - renvoie Tensor(new_shape, new_strides, data_) -- le meme
//   constructeur prive que transpose() utilise, `data_` partage tel
//   quel, aucune copie.
Tensor Tensor::reshape(std::vector<std::size_t> new_shape) const {
    std::size_t new_total_size = std::accumulate(new_shape.begin(), new_shape.end(), std::size_t{1}, std::multiplies<std::size_t>());
    if (new_total_size != size()) {
        throw std::invalid_argument("New shape must have the same total size as the original tensor");
    }
    std::vector<std::size_t> new_strides(new_shape.size());
    new_strides[new_shape.size() - 1] = 1;
    for (std::size_t k = new_shape.size() - 1; k > 0; --k) {
        new_strides[k - 1] = new_strides[k] * new_shape[k];
    }
    return Tensor(new_shape, new_strides, data_);
}

// TODO(toi), etape par etape -- la plus dense du module, prends ton
// temps. Exemple travaille pour t'orienter : ce Tensor a shape_ =
// {3} (donc strides_ = {1}), target_shape = {2, 3} (ajouter le meme
// vecteur de 3 nombres a chaque ligne d'une matrice 2x3).
//
// - this_ndim = ndim() (ici 1), target_ndim = target_shape.size()
//   (ici 2). Si this_ndim > target_ndim, throw std::invalid_argument
//   (on ne peut pas "retrecir" le nombre de dimensions avec
//   broadcast_to).
//
// - offset = target_ndim - this_ndim (ici 2 - 1 = 1) : le nombre de
//   dimensions implicites de taille 1 ajoutees au debut.
//
// - construis new_strides, un vector<size_t> de taille target_ndim.
//   Pour chaque k de 0 a target_ndim - 1 :
//     - si k < offset (ici, seulement k=0) : dimension implicite,
//       new_strides[k] = 0 (rien a comparer, elle "vaut" 1 par
//       convention et s'etire toujours)
//     - sinon : orig_dim = k - offset (ici pour k=1, orig_dim = 0).
//       Compare shape_[orig_dim] (ici shape_[0] = 3) a
//       target_shape[k] (ici target_shape[1] = 3) :
//         - si egales : new_strides[k] = strides_[orig_dim] (pas
//           d'etirement, comportement normal -- ici
//           new_strides[1] = strides_[0] = 1)
//         - si shape_[orig_dim] == 1 (mais pas egal a
//           target_shape[k]) : new_strides[k] = 0 (etirement)
//         - sinon : incompatible, throw std::invalid_argument
//
//   Resultat attendu sur l'exemple : new_strides = {0, 1}. Verifie
//   avec flat_index : {0,j} -> 0*0 + j*1 = j, et {1,j} -> 1*0 + j*1 = j
//   -- les deux "lignes" lisent bien les 3 memes valeurs.
//
// - renvoie Tensor(target_shape, new_strides, data_)
Tensor Tensor::broadcast_to(std::vector<std::size_t> target_shape) const {
    std::size_t this_ndim = ndim();
    std::size_t target_ndim = target_shape.size();
    if (this_ndim > target_ndim) {
        throw std::invalid_argument("Cannot broadcast to a shape with fewer dimensions");
    }
    std::size_t offset = target_ndim - this_ndim;
    std::vector<std::size_t> new_strides(target_ndim);
    for (std::size_t k = 0; k < target_ndim; ++k) {
        if (k < offset) {
            new_strides[k] = 0;
        } else {
            std::size_t orig_dim = k - offset;
            if (shape_[orig_dim] == target_shape[k]) {
                new_strides[k] = strides_[orig_dim];
            } else if (shape_[orig_dim] == 1) {
                new_strides[k] = 0;
            } else {
                throw std::invalid_argument("Incompatible shapes for broadcasting");
            }
        }
    }
    return Tensor(target_shape, new_strides, data_);
}

} // namespace chiikaml
