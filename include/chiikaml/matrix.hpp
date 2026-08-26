#pragma once

#include <cstddef>
#include <iostream>
#include <vector>

namespace chiikaml {

// Matrice dense de doubles, stockee de maniere contigue en memoire
// (row-major : la ligne i, colonne j est a l'indice i * cols_ + j).
//
// Pourquoi pas std::vector<std::vector<double>> ?
// -> Ce serait un vecteur de vecteurs : chaque ligne est allouee
//    separement, dispersee dans le tas. Mauvais pour le cache CPU.
//    Un seul bloc contigu est beaucoup plus rapide a parcourir.
class Matrix {
public:
    // Construit une matrice rows x cols, initialisee a 0.0.
    Matrix(std::size_t rows, std::size_t cols);

    std::size_t rows() const;
    std::size_t cols() const;

    // Accès lecture/écriture à l'élément (row, col).
    double& operator()(std::size_t row, std::size_t col);
    double operator()(std::size_t row, std::size_t col) const;

    // Addition élément par élément. Les deux matrices doivent avoir
    // les mêmes dimensions.
    Matrix operator+(const Matrix& other) const;

    // Change les dimensions vers rows x cols. Les valeurs existantes
    // dans la zone commune aux anciennes et nouvelles dimensions
    // (min(rows(), rows) x min(cols(), cols)) sont préservées, à la
    // bonne case (row, col) dans les deux cas. Toute case
    // nouvellement créée (agrandissement) vaut 0.0. Toute case en
    // dehors des nouvelles dimensions (rétrécissement) est perdue.
    //
    // Attention : contrairement à std::vector::resize, on ne peut pas
    // se contenter de redimensionner data_ tel quel -- le stockage
    // est row-major avec un "pas" (le nombre de colonnes) qui change
    // potentiellement lui aussi. Il faut donc reconstruire un nouveau
    // buffer et recopier case par case, en utilisant l'ANCIEN pas
    // (cols()) pour lire dans l'ancien buffer et le NOUVEAU pas
    // (cols, le paramètre) pour écrire dans le nouveau.
    void resize(std::size_t rows, std::size_t cols);

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;
};

std::ostream& operator<<(std::ostream& os, const Matrix& m);

} // namespace chiikaml
