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
    throw std::logic_error("Tensor::operator() pas encore implemente");
}

double Tensor::operator()(const std::vector<std::size_t>& indices) const {
    throw std::logic_error("Tensor::operator() const pas encore implemente");
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
    throw std::logic_error("Tensor::operator+ pas encore implemente");
}

} // namespace chiikaml
