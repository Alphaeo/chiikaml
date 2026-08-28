#pragma once

#include <cstddef>
#include <iostream>
#include <string>
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

    // Definies directement dans la classe (donc implicitement
    // `inline`) plutot que dans matrix.cpp : ce sont des accesseurs
    // triviaux, appeles dans les boucles chaudes de tous les autres
    // modules (KNNClassifier, KDTree, KMeans, RandomForest...).
    // Si elles vivaient dans matrix.cpp, le compilateur ne pourrait
    // pas "voir a travers" l'appel depuis un autre fichier .cpp
    // (une autre unite de compilation) -- il devrait supposer par
    // prudence qu'elles peuvent modifier n'importe quoi en memoire,
    // ce qui bloque l'auto-vectorisation des boucles qui les
    // appellent. Verifie empiriquement avec `-fopt-info-vec-all` :
    // avant ce changement, GCC refusait de vectoriser la boucle de
    // KNNClassifier::squared_distance() pour exactement cette raison.
    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    // Accès lecture/écriture à l'élément (row, col).
    double& operator()(std::size_t row, std::size_t col) { return data_[row * cols_ + col]; }
    double operator()(std::size_t row, std::size_t col) const { return data_[row * cols_ + col]; }

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

    // Charge une matrice depuis un fichier CSV : une ligne de texte
    // = une ligne de la matrice, des valeurs numeriques separees par
    // des virgules. Pas d'en-tete de colonnes suppose -- toutes les
    // lignes du fichier doivent etre numeriques. Leve
    // std::runtime_error si le fichier ne peut pas etre ouvert,
    // std::invalid_argument si les lignes n'ont pas toutes le meme
    // nombre de colonnes.
    //
    // static : pas besoin d'une Matrix deja existante pour appeler
    // ca -- on l'appelle directement sur la classe, comme
    // Matrix::from_csv("data.csv"), pas m.from_csv(...).
    static Matrix from_csv(const std::string& path);

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;
};

std::ostream& operator<<(std::ostream& os, const Matrix& m);

} // namespace chiikaml
