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

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;
};

std::ostream& operator<<(std::ostream& os, const Matrix& m);

} // namespace chiikaml
