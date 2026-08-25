#pragma once

#include <cstddef>
#include <random>
#include <vector>

#include "chiikaml/matrix.hpp"

namespace chiikaml {

// Clustering par K-Means (algorithme de Lloyd) : partitionne les
// lignes de X en n_clusters groupes, en alternant deux etapes :
//  - affectation : chaque point rejoint le centre (centroide) le
//    plus proche
//  - mise a jour : chaque centre devient la moyenne des points qui
//    lui sont actuellement assignes
// jusqu'a ce que plus aucune affectation ne change (convergence) ou
// que max_iterations soit atteint (garde-fou : Lloyd converge presque
// toujours vite en pratique, mais rien ne garantit une borne stricte).
//
// Les centres initiaux sont tires au sort (sans remise) parmi les
// points de X -- volontairement simple pour cette premiere version.
// C'est une limite connue : un mauvais tirage initial peut faire
// converger vers un optimum local mediocre (ex: deux centres tires
// dans le meme vrai cluster). Une amelioration classique,
// k-means++, choisit les centres initiaux pour qu'ils soient deja
// eloignes les uns des autres -- pas implementee ici, mais a garder
// en tete comme piste d'amelioration.
//
// `seed` rend le tirage reproductible : memes donnees + meme seed =
// meme resultat, a chaque execution. Utile pour les tests, et plus
// generalement une bonne pratique en ML (une experience qu'on ne
// peut pas reproduire est difficile a deboguer ou a comparer).
class KMeans {
public:
    explicit KMeans(std::size_t n_clusters, std::size_t max_iterations = 100,
                     unsigned int seed = 42);

    // Calcule les centres a partir des points de X (une ligne = un
    // point). Modifie centroids_ et labels_.
    void fit(const Matrix& X);

    // Assigne chaque ligne de X au centre le plus proche parmi ceux
    // trouves par fit() (doit etre appelee apres fit()). Contrairement
    // a fit(), ne modifie jamais les centres : sert a classer de
    // nouveaux points par rapport a un clustering deja calcule.
    std::vector<std::size_t> predict(const Matrix& X) const;

    // Le cluster assigne a chaque point d'entrainement (valide apres fit()).
    const std::vector<std::size_t>& labels() const;

    // Les n_clusters centres trouves (valide apres fit()).
    const Matrix& centroids() const;

private:
    std::size_t n_clusters_;
    std::size_t max_iterations_;
    std::mt19937 rng_;

    Matrix centroids_;
    std::vector<std::size_t> labels_;

    // Indice du centre (dans centroids_) le plus proche de la ligne
    // `row` de `points`. Utilisee a la fois par fit() (etape
    // d'affectation) et predict() -- la meme operation au fond.
    std::size_t nearest_centroid(const Matrix& points, std::size_t row) const;

    // Ne depend d'aucun etat de KMeans (pas de centroids_, pas de
    // rng_) : les deux points compares sont entierement donnes par
    // les parametres. Une fonction qui n'a pas besoin de `this` peut
    // etre declaree `static` -- elle appartient toujours a la classe
    // (namespacee comme KMeans::squared_distance), mais s'appelle
    // sans instance et ne peut pas acceder aux membres non-statiques.
    static double squared_distance(const Matrix& a, std::size_t row_a, const Matrix& b,
                                    std::size_t row_b);
};

} // namespace chiikaml
