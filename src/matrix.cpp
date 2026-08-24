#include "chiikaml/matrix.hpp"

#include <stdexcept>

namespace chiikaml {

// TODO(toi): implemente ce constructeur.
// - stocke rows/cols
// - initialise data_ avec la bonne taille (rows * cols), rempli de 0.0
Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols) {

    std::size_t size = rows * cols;
    data_.resize(size, 0.0);

}

std::size_t Matrix::rows() const {
    
    return rows_;

}

std::size_t Matrix::cols() const {
    
    return cols_;

}

// TODO(toi): renvoie une reference vers data_[row * cols_ + col]
double& Matrix::operator()(std::size_t row, std::size_t col) {

    return data_[row * cols_ + col];
}

double Matrix::operator()(std::size_t row, std::size_t col) const {
    
    return data_[row * cols_ + col];
}

// TODO(toi): cree une matrice resultat de meme taille, additionne
// element par element.
Matrix Matrix::operator+(const Matrix& other) const {

    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrices must have the same dimensions");
    }

    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            result(i, j) = (*this)(i, j) + other(i, j);
        }
    }
    return result;
}

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    
    for (std::size_t i=0; i<m.rows(); ++i){
        for (std::size_t j=0; j<m.cols(); ++j){
            os << m(i,j) << ' ';
        }
        os << '\n';
    }

    return os;
}

} // namespace chiikaml
