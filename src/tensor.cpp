#include "chiikaml/tensor.hpp"

#include <numeric>
#include <stdexcept>
#include <algorithm>

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

namespace {

// Incremente `index` d'un cran par rapport a `shape`, facon compteur
// kilometrique (odometer) : c'est le meme principe qu'une addition
// avec retenue, mais en "base shape[k]" a chaque position k au lieu
// d'une base fixe (10 pour un compteur decimal classique).
//
// TODO(toi), etape par etape :
//
// - parcours les positions de `index` A L'ENVERS (de la DERNIERE vers
//   la premiere -- utilise un indice signe, ou compte a rebours, pas
//   les reverse iterators ici puisqu'on doit pouvoir s'arreter au
//   milieu).
//
// - pour chaque position k (en partant de la fin) : incremente
//   index[k]. Si index[k] < shape[k] apres l'incrementation, tu as
//   fini -- renvoie true (pas de retenue a propager plus loin).
//
// - sinon (index[k] a atteint shape[k], "deborde") : remets index[k]
//   a 0 (retenue) et continue vers la position PRECEDENTE (k-1) --
//   exactement comme "9+1" devient "0" avec une retenue de 1 vers la
//   colonne suivante en addition decimale.
//
// - si tu deborde meme sur la toute premiere position (k==0 a
//   deborde) : il n'y a plus de position ou propager la retenue,
//   toutes les combinaisons ont ete parcourues -- renvoie false.
//
// - cas particulier a ne pas oublier : shape vide (index vide, tenseur
//   sans dimension de lot) -- il n'y a qu'une seule "combinaison"
//   (l'index vide lui-meme), donc un seul appel a ce point de vue
//   devrait renvoyer false immediatement (rien a incrementer).
bool increment_index(std::vector<std::size_t>& index, const std::vector<std::size_t>& shape) {
    std::size_t ndim = shape.size();
    for (std::size_t k = ndim; k-- > 0;) {
        index[k]++;
        if (index[k] < shape[k]) {
            return true;
        } else {
            index[k] = 0;
        }
    }
    return false;
}

} // namespace

// TODO(toi), etape par etape -- la plus grosse fonction de Tensor,
// mais chaque etape reutilise quelque chose que tu as deja ecrit :
//
// - verifie ce->ndim() >= 2 ET other.ndim() >= 2, sinon throw
//   std::invalid_argument.
//
// - extrait les dimensions matricielles : m = shape_[ndim()-2],
//   k = shape_[ndim()-1], k2 = other.shape()[other.ndim()-2],
//   n = other.shape()[other.ndim()-1]. Si k != k2, throw
//   std::invalid_argument.
//
// - calcule batch_shape : la combinaison (façon broadcast_to) des
//   dimensions de lot de ce tenseur (shape_ SANS les deux dernieres)
//   et de other (other.shape() SANS les deux dernieres). Meme
//   principe que la boucle de broadcast_to(), mais entre DEUX formes
//   au lieu d'une forme et une cible fixe :
//     - batch_ndim = max(ndim()-2, other.ndim()-2)
//     - pour k de 0 a batch_ndim-1 (en partant de la DROITE, comme
//       dans broadcast_to) : recupere dim_a (1 implicite si ce
//       tenseur n'a pas assez de dimensions de lot a cette position)
//       et dim_b (meme logique) ; si dim_a == dim_b -> garde cette
//       valeur ; sinon si l'un des deux vaut 1 -> prends l'autre ;
//       sinon -> throw std::invalid_argument.
//
// - construis les formes cibles pour chaque tenseur : batch_shape +
//   {m, k} pour ce tenseur, batch_shape + {k, n} pour other. Utilise
//   broadcast_to() (deja ecrit !) sur chacun pour obtenir deux VUES
//   qui ont maintenant exactement la meme forme de lot.
//
// - cree Tensor result(batch_shape + {m, n}) (deja a 0.0).
//
// - parcours TOUTES les combinaisons d'indices de lot : un
//   std::vector<std::size_t> batch_index(batch_shape.size(), 0), puis
//   une boucle "do { ... } while (increment_index(batch_index,
//   batch_shape));" -- a chaque iteration, fais la triple boucle
//   matricielle classique (i, j, p) en construisant les indices
//   complets par CONCATENATION : batch_index + {i, p} pour lire dans
//   la vue broadcastee de ce tenseur, batch_index + {p, j} pour
//   other, batch_index + {i, j} pour ecrire dans result.
//
// - renvoie result.
Tensor Tensor::matmul(const Tensor& other) const {
    
    if (ndim() < 2 || other.ndim() < 2) {
        throw std::invalid_argument("Both tensors must have at least 2 dimensions for matmul");
    }

    std::size_t m = shape_[ndim() - 2];
    std::size_t k = shape_[ndim() - 1];
    std::size_t k2 = other.shape()[other.ndim() - 2];
    std::size_t n = other.shape()[other.ndim() - 1];

    if (k != k2) {
        throw std::invalid_argument("Inner dimensions must match for matmul");
    }

    // Compute batch_shape
    std::vector<std::size_t> batch_shape;
    std::size_t this_batch_ndim = ndim() - 2;
    std::size_t other_batch_ndim = other.ndim() - 2;
    std::size_t batch_ndim = std::max(this_batch_ndim, other_batch_ndim);

    for (std::size_t i = 0; i < batch_ndim; ++i) {
        std::size_t dim_a = (i < this_batch_ndim) ? shape_[this_batch_ndim - 1 - i] : 1;
        std::size_t dim_b = (i < other_batch_ndim) ? other.shape()[other_batch_ndim - 1 - i] : 1;

        if (dim_a == dim_b) {
            batch_shape.push_back(dim_a);
        } else if (dim_a == 1) {
            batch_shape.push_back(dim_b);
        } else if (dim_b == 1) {
            batch_shape.push_back(dim_a);
        } else {
            throw std::invalid_argument("Incompatible batch dimensions for matmul");
        }
    }
    std::reverse(batch_shape.begin(), batch_shape.end());

    // Broadcast tensors to the same batch shape
    std::vector<size_t> size_a = batch_shape;
    size_a.push_back(m);
    size_a.push_back(k);
    std::vector<size_t> size_b = batch_shape;
    size_b.push_back(k);
    size_b.push_back(n);

    Tensor a_broadcasted = broadcast_to(size_a);
    Tensor b_broadcasted = other.broadcast_to(size_b);

    // Create result tensor
    std::vector<std::size_t> result_shape = batch_shape;
    result_shape.push_back(m);
    result_shape.push_back(n);
    Tensor result(result_shape);

    // Iterate over all combinations of batch indices. Contrairement a
    // la version precedente, on n'alloue plus aucun vector<size_t>
    // dans les boucles i/j/p : on descend directement au niveau des
    // strides_ et des buffers plats (comme operator+), pour eviter
    // une allocation heap par element -- l'allocation dominait
    // largement le cout reel du calcul (une addition/multiplication
    // est bien moins chere qu'un malloc).
    double* a_data = a_broadcasted.data_->data();
    double* b_data = b_broadcasted.data_->data();
    double* result_data = result.data_->data();

    // Strides de la partie matricielle (les 2 dernieres dimensions),
    // identiques pour toute combinaison d'indices de lot.
    std::size_t a_row_stride = a_broadcasted.strides_[batch_ndim];       // avancer de 1 sur i
    std::size_t a_col_stride = a_broadcasted.strides_[batch_ndim + 1];   // avancer de 1 sur p
    std::size_t b_row_stride = b_broadcasted.strides_[batch_ndim];       // avancer de 1 sur p
    std::size_t b_col_stride = b_broadcasted.strides_[batch_ndim + 1];   // avancer de 1 sur j
    std::size_t result_row_stride = result.strides_[batch_ndim];        // avancer de 1 sur i
    std::size_t result_col_stride = result.strides_[batch_ndim + 1];    // avancer de 1 sur j

    std::vector<std::size_t> batch_index(batch_shape.size(), 0);
    do {
        // Offset de depart pour cette combinaison de lot -- calcule
        // UNE fois par combinaison (cout O(batch_ndim)), pas a chaque
        // element de la matrice (cout O(m*n*k) sinon).
        std::size_t a_batch_offset = 0;
        std::size_t b_batch_offset = 0;
        std::size_t result_batch_offset = 0;
        for (std::size_t d = 0; d < batch_shape.size(); ++d) {
            a_batch_offset += batch_index[d] * a_broadcasted.strides_[d];
            b_batch_offset += batch_index[d] * b_broadcasted.strides_[d];
            result_batch_offset += batch_index[d] * result.strides_[d];
        }

        for (std::size_t i = 0; i < m; ++i) {
            std::size_t a_row_offset = a_batch_offset + i * a_row_stride;
            std::size_t result_row_offset = result_batch_offset + i * result_row_stride;

            for (std::size_t j = 0; j < n; ++j) {
                std::size_t b_col_offset = b_batch_offset + j * b_col_stride;

                double sum = 0.0;
                for (std::size_t p = 0; p < k; ++p) {
                    sum += a_data[a_row_offset + p * a_col_stride] * b_data[b_col_offset + p * b_row_stride];
                }
                result_data[result_row_offset + j * result_col_stride] = sum;
            }
        }
    } while (increment_index(batch_index, batch_shape));

    return result;
}

} // namespace chiikaml
