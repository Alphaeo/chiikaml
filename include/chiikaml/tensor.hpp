#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace chiikaml {

// Tenseur dense de doubles, a N dimensions -- la generalisation de
// Matrix (qui n'en avait que 2, fixees : rows_/cols_). Stockage
// toujours contigu dans un buffer plat (meme philosophie que
// Matrix), mais avec deux nouveautes :
//
//  - shape_ : la taille de chaque dimension (ex : {2, 3, 4} pour un
//    tenseur 2x3x4). Un vector, pas deux size_t fixes -- Tensor doit
//    marcher pour N'IMPORTE QUEL nombre de dimensions.
//
//  - strides_ : combien d'elements sauter dans le buffer plat pour
//    avancer d'UN cran sur chaque dimension. Pour un tenseur
//    contigu "normal", strides_[k] = produit des tailles de toutes
//    les dimensions APRES k (pour Matrix, cols_ jouait exactement ce
//    role, implicitement, pour la seule dimension qui comptait).
//    Le vrai interet des strides, pour plus tard (pas encore dans
//    cette version) : ils permettent des VUES sans copie -- une
//    transposition ou un reshape n'auraient qu'a changer shape_/
//    strides_, jamais recopier les nombres eux-memes.
//
// data_ est un shared_ptr (pas un vector directement, contrairement
// a Matrix) : ce choix prepare le terrain pour que plusieurs Tensor
// (par exemple un tenseur et une vue transposee de lui-meme, plus
// tard) puissent partager le MEME buffer sous-jacent sans jamais
// copier les donnees -- seuls shape_/strides_ different. Pas encore
// exploite dans cette version (chaque Tensor cree et possede son
// propre buffer), mais le design est deja pret pour ca.
//
// Limitation assumee de cette v1 : pas de matmul, pas de
// broadcasting, pas de transposition/reshape, pas de SIMD -- juste
// la structure de base (forme, strides, acces, addition) sur
// laquelle tout le reste de la Phase 9 va s'empiler par etapes,
// exactement comme Matrix a grandi phase par phase pendant toute la
// Partie 1.
class Tensor {
public:
    // Construit un tenseur de la forme donnee, initialise a 0.0.
    // Ex : Tensor({2, 3}) est l'equivalent d'un Matrix(2, 3) ;
    // Tensor({2, 3, 4}) n'a pas d'equivalent Matrix, seul Tensor le
    // permet.
    explicit Tensor(std::vector<std::size_t> shape);

    const std::vector<std::size_t>& shape() const { return shape_; }

    // Nombre de dimensions (2 pour un Tensor("comme une Matrix"), 3
    // pour {2,3,4}, etc.)
    std::size_t ndim() const { return shape_.size(); }

    // Nombre total d'elements (produit de toutes les dimensions).
    std::size_t size() const;

    // Acces lecture/ecriture a l'element dont les indices sont
    // donnes -- un indice par dimension, dans l'ordre de shape()
    // (ex : pour un Tensor({2,3,4}), operator()({1, 2, 0}) acceder
    // a la ligne 1, colonne 2, profondeur 0).
    double& operator()(const std::vector<std::size_t>& indices);
    double operator()(const std::vector<std::size_t>& indices) const;

    // Addition element par element. Les deux tenseurs doivent avoir
    // exactement la meme forme (pas de broadcasting pour cette
    // premiere version -- viendra plus tard dans la Phase 9).
    Tensor operator+(const Tensor& other) const;

    // Renvoie une VUE transposee de ce tenseur : shape_ et strides_
    // inverses (pour un tenseur 2D, c'est la transposition classique
    // -- lignes et colonnes echangees ; pour un tenseur a N
    // dimensions, c'est l'ordre des dimensions entierement inverse).
    //
    // AUCUNE donnee n'est copiee : la vue renvoyee partage exactement
    // le meme buffer data_ que l'original (c'est precisement pour ca
    // que data_ est un shared_ptr). Ecrire dans la vue transposee
    // modifie donc AUSSI l'original -- c'est le comportement attendu
    // d'une vue, pas un bug.
    Tensor transpose() const;

    // Renvoie une VUE de ce tenseur avec une nouvelle forme, tant que
    // le nombre total d'elements reste identique (ex : {2, 6} ->
    // {3, 4}, ou {12} -- les memes 12 nombres, juste regroupes
    // differemment). Comme transpose(), AUCUNE donnee n'est copiee.
    // Leve std::invalid_argument si le nombre total d'elements de
    // new_shape ne correspond pas a size().
    //
    // Limitation assumee de cette version : suppose que ce Tensor est
    // deja CONTIGU en memoire. C'est le cas de tout Tensor construit
    // normalement (via le constructeur public), mais PAS forcement le
    // cas d'une vue deja transposee -- appeler reshape() sur le
    // resultat de transpose() n'est pas garanti correct dans cette
    // v1 (a revisiter plus tard si besoin).
    Tensor reshape(std::vector<std::size_t> new_shape) const;

    // Renvoie une VUE de ce tenseur "etiree" vers target_shape, selon
    // les regles de broadcasting classiques (NumPy) : les formes sont
    // comparees en partant de la DROITE. Chaque dimension doit etre
    // soit egale a la dimension cible, soit valoir 1 dans CE tenseur
    // (auquel cas elle est "etiree" pour correspondre). Si ce tenseur
    // a moins de dimensions que target_shape, des dimensions
    // implicites de taille 1 sont ajoutees au DEBUT.
    //
    // AUCUNE donnee n'est copiee ni dupliquee : une dimension etiree
    // recoit un STRIDE DE 0 -- avancer sur cette dimension ne bouge
    // jamais dans le buffer, donc n'importe quel indice sur cette
    // dimension lit la MEME case memoire sous-jacente.
    //
    // ATTENTION : consequence de ce stride 0 -- ECRIRE dans une vue
    // broadcastee via un indice non nul sur une dimension etiree
    // modifierait la MEME case que l'indice 0 sur cette dimension
    // (elles pointent vers le meme endroit). Cette vue est pensee
    // pour la LECTURE (les futures operations comme + qui
    // l'utiliseront en interne), pas pour l'ecriture directe.
    //
    // Leve std::invalid_argument si target_shape a moins de
    // dimensions que ce tenseur, ou si une dimension n'est ni egale
    // ni egale a 1 (aucune regle de broadcasting ne s'applique).
    Tensor broadcast_to(std::vector<std::size_t> target_shape) const;

private:
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::shared_ptr<std::vector<double>> data_;

    // Constructeur "prive" pour construire une VUE : pas de nouvelle
    // allocation, juste un nouveau shape_/strides_ par-dessus un
    // data_ DEJA existant (partage, pas copie -- copier un shared_ptr
    // est "cheap" : ca incremente juste un compteur de references).
    // Utilise par transpose() (et plus tard reshape()). Trivial
    // (juste une init de membres), donc deja ecrit ici.
    Tensor(std::vector<std::size_t> shape, std::vector<std::size_t> strides,
           std::shared_ptr<std::vector<double>> data)
        : shape_(std::move(shape)), strides_(std::move(strides)), data_(std::move(data)) {}

    // Traduit des indices multi-dimensionnels en un seul indice dans
    // le buffer plat data_, via un produit scalaire indices . strides_.
    std::size_t flat_index(const std::vector<std::size_t>& indices) const;
};

} // namespace chiikaml
