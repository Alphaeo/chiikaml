#include "chiikaml/matrix.hpp"

#include <stdexcept>
#include <algorithm>

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

// TODO(toi), etape par etape :
// - garde rows_ et cols_ actuels quelque part (ou lis-les avant de
//   les ecraser) -- ce sont l' "ancien pas" pour lire data_
// - cree un nouveau std::vector<double> new_data(rows * cols, 0.0)
// - common_rows = min(rows_, rows), common_cols = min(cols_, cols)
// - pour i dans [0, common_rows), pour j dans [0, common_cols) :
//   new_data[i * cols + j] = data_[i * cols_ + j]
//   (bien remarquer : cols pour ecrire dans new_data, cols_ -- le
//   membre, donc l'ancien -- pour lire dans data_)
// - rows_ = rows; cols_ = cols; data_ = std::move(new_data);
void Matrix::resize(std::size_t rows, std::size_t cols) {
    
    std::size_t old_rows = rows_;
    std::size_t old_cols = cols_;
    std::vector<double> new_data(rows * cols, 0.0);

    std::size_t common_rows = std::min(old_rows, rows);
    std::size_t common_cols = std::min(old_cols, cols);

    for (std::size_t i = 0; i < common_rows; ++i) {
        for (std::size_t j = 0; j < common_cols; ++j) {
            new_data[i * cols + j] = data_[i * old_cols + j];
        }
    }

    rows_ = rows;
    cols_ = cols;
    data_ = std::move(new_data);
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
