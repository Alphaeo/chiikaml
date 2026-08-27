#pragma once

#include "chiikaml/kmeans.hpp"
#include "chiikaml/matrix.hpp"

namespace chiikaml::viz {

// Ouvre (ou met a jour, si deja ouvert) le dashboard local avec un
// nuage de points representant le clustering de `model` sur `X`
// (les points colores par cluster, les centres marques d'une croix).
//
// Limitation de cette v1 : seulement les donnees 2D (X.cols() == 2)
// sont supportees -- un nuage de points n'a de sens visuel direct
// qu'en 2D. Leve std::invalid_argument sinon.
//
// Volontairement une fonction libre plutot qu'une methode
// KMeans::visualize() : ca eviterait sinon que le coeur de KMeans
// (kmeans.hpp/kmeans.cpp, dans la bibliotheque chiikaml de base)
// depende de nlohmann::json et httplib -- meme principe que les
// bindings Python, garder le coeur C++ independant de tout
// dependance optionnelle.
void visualize(const KMeans& model, const Matrix& X);

} // namespace chiikaml::viz
