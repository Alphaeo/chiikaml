#include "chiikaml/matrix.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace chiikaml {

// TODO(toi): implemente ce constructeur.
// - stocke rows/cols
// - initialise data_ avec la bonne taille (rows * cols), rempli de 0.0
Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols) {

    std::size_t size = rows * cols;
    data_.resize(size, 0.0);

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

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument(
            "Le nombre de colonnes de la premiere matrice "
            "doit etre egal au nombre de lignes de la seconde"
        );
    }

    Matrix result(rows_, other.cols_); // create a result matrix with the correct dimensions

    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t k = 0; k < cols_; ++k) {
            const double a_ik = (*this)(i, k);

            for (std::size_t j = 0; j < other.cols_; ++j) {
                result(i, j) += a_ik * other(k, j);
            }
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

// TODO(toi), etape par etape :
//
// - std::ifstream file(path); si !file.is_open(), leve
//   std::runtime_error("impossible d'ouvrir " + path) (ou message
//   similaire)
//
// - un std::vector<std::vector<double>> rows_parsed pour accumuler
//   les lignes au fur et a mesure (on ne connait pas encore le
//   nombre total de lignes/colonnes a l'avance, donc pas possible de
//   construire directement une Matrix)
//
// - std::string line; tant que std::getline(file, line) reussit
//   (chaque appel lit une ligne du fichier dans `line`, renvoie
//   false quand il n'y a plus rien a lire -- utilisable directement
//   comme condition de boucle "while (std::getline(file, line))") :
//     - decoupe `line` sur les virgules. Une facon simple : un
//       std::stringstream construit a partir de `line`, puis
//       std::getline(ce_stringstream, token, ',') dans une boucle
//       (le troisieme argument de getline est le caractere
//       separateur -- par defaut c'est '\n', ici on veut ',')
//     - pour chaque token (un bout de texte entre deux virgules),
//       convertis-le en double avec std::stod(token) et ajoute-le a
//       un std::vector<double> pour cette ligne
//     - une fois la ligne entierement decoupee, push_back ce
//       vector<double> dans rows_parsed
//
// - verifie que rows_parsed n'est pas vide, et que CHAQUE ligne de
//   rows_parsed a le meme nombre de colonnes que la premiere (sinon
//   throw std::invalid_argument)
//
// - cree Matrix result(rows_parsed.size(), rows_parsed[0].size()),
//   copie rows_parsed dedans (deux boucles i/j classiques), renvoie
//   result
Matrix Matrix::from_csv(const std::string& path) {
    
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("impossible d'ouvrir " + path);
    }

    std::vector<std::vector<double>> rows_parsed;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> row;
        while (std::getline(ss, token, ',')) {
            row.push_back(std::stod(token));
        }
        rows_parsed.push_back(row);
    }

    if (rows_parsed.empty()) {
        throw std::invalid_argument("fichier CSV vide");
    }

    std::size_t num_cols = rows_parsed[0].size();
    for (const auto& row : rows_parsed) {
        if (row.size() != num_cols) {
            throw std::invalid_argument("lignes de longueur variable");
        }
    }

    Matrix result(rows_parsed.size(), num_cols);
    for (std::size_t i = 0; i < rows_parsed.size(); ++i) {
        for (std::size_t j = 0; j < num_cols; ++j) {
            result(i, j) = rows_parsed[i][j];
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

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument(
            "Matrices must have the same dimensions"
        );
    }

    return *this + (other * -1.0);
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows_, cols_);

    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }

    return result;
}

Matrix Matrix::operator/(double scalar) const {
    if (scalar == 0.0) {
        throw std::invalid_argument("Division by zero");
    }

    Matrix result(rows_, cols_);

    // Computing the reciprocal once avoids performing one division
    // for every matrix coefficient.
    const double reciprocal = 1.0 / scalar;

    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * reciprocal;
    }

    return result;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument(
            "Matrices must have the same dimensions"
        );
    }

    return *this = *this + other;
}


Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument(
            "Matrices must have the same dimensions"
        );
    }

    return *this = *this - other;
}

Matrix& Matrix::operator*=(double scalar) {
    for (double& value : data_) {
        value *= scalar;
    }

    return *this;
}

// Scalar multiplication from the left: scalar * matrix.
Matrix operator*(double scalar, const Matrix& matrix) {
    return matrix * scalar;
}

// Computes the determinant of a square matrix.
// Throws std::invalid_argument if the matrix is not square.
double Matrix::det() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("The matrix must be square");
    }

    const std::size_t n = rows_;

    // By convention, the determinant of a 0x0 matrix is 1.
    if (n == 0) {
        return 1.0;
    }

    // Create a copy because Gaussian elimination modifies the coefficients.
    std::vector<double> lu = data_;
    int sign = 1;

    for (std::size_t k = 0; k < n; ++k) {
        // Find the largest pivot in column k for partial pivoting.
        std::size_t pivot_row = k;
        double max_value = std::abs(lu[k * n + k]);

        for (std::size_t i = k + 1; i < n; ++i) {
            const double value = std::abs(lu[i * n + k]);

            if (value > max_value) {
                max_value = value;
                pivot_row = i;
            }
        }

        // If no nonzero pivot exists, the matrix is singular.
        if (max_value == 0.0) {
            return 0.0;
        }

        // Move the selected pivot onto the diagonal.
        if (pivot_row != k) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(
                    lu[k * n + j],
                    lu[pivot_row * n + j]
                );
            }

            // Swapping two rows changes the sign of the determinant.
            sign = -sign;
        }

        const double pivot = lu[k * n + k];

        // Eliminate the coefficients below the pivot.
        for (std::size_t i = k + 1; i < n; ++i) {
            const double factor = lu[i * n + k] / pivot;
            lu[i * n + k] = 0.0;

            for (std::size_t j = k + 1; j < n; ++j) {
                lu[i * n + j] -= factor * lu[k * n + j];
            }
        }
    }

    double determinant = static_cast<double>(sign);

    for (std::size_t i = 0; i < n; ++i) {
        determinant *= lu[i * n + i];
    }

    return determinant;
}

// Computes the inverse of a square matrix using LU decomposition. (doesn't require previous computation of the determinant)
Matrix Matrix::inv() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("The matrix must be square");
    }

    const std::size_t n = rows_;

    // The inverse of the empty matrix is the empty matrix.
    if (n == 0) {
        return Matrix(0, 0);
    }

    // Store both L and U in the same contiguous buffer:
    // - U is stored on and above the diagonal.
    // - L is stored below the diagonal.
    // - The diagonal of L is implicitly equal to 1.
    std::vector<double> lu = data_;

    // pivots[k] records the row swapped with row k during factorization.
    std::vector<std::size_t> pivots(n);

    // Compute the LU factorization with partial pivoting.
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot_row = k;
        double max_value = std::abs(lu[k * n + k]);

        for (std::size_t i = k + 1; i < n; ++i) {
            const double value = std::abs(lu[i * n + k]);

            if (value > max_value) {
                max_value = value;
                pivot_row = i;
            }
        }

        // No nonzero pivot means that the matrix is singular.
        if (max_value == 0.0) {
            throw std::runtime_error("The matrix is singular");
        }

        pivots[k] = pivot_row;

        // Move the selected pivot onto the diagonal.
        if (pivot_row != k) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(
                    lu[k * n + j],
                    lu[pivot_row * n + j]
                );
            }
        }

        const double pivot = lu[k * n + k];

        // Compute the multipliers of L and update the remaining
        // submatrix to construct U.
        for (std::size_t i = k + 1; i < n; ++i) {
            lu[i * n + k] /= pivot;
            const double multiplier = lu[i * n + k];

            for (std::size_t j = k + 1; j < n; ++j) {
                lu[i * n + j] -= multiplier * lu[k * n + j];
            }
        }
    }

    // Start with the identity matrix. It represents the right-hand
    // side of AX = I.
    Matrix inverse(n, n);

    for (std::size_t i = 0; i < n; ++i) {
        inverse(i, i) = 1.0;
    }

    // Apply to the identity matrix the same row permutations that
    // were applied during the LU factorization.
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t pivot_row = pivots[k];

        if (pivot_row != k) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(
                    inverse(k, j),
                    inverse(pivot_row, j)
                );
            }
        }
    }

    // Forward substitution: solve LY = P * I.
    // The diagonal coefficients of L are implicitly equal to 1.
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < i; ++k) {
            const double l_ik = lu[i * n + k];

            for (std::size_t j = 0; j < n; ++j) {
                inverse(i, j) -= l_ik * inverse(k, j);
            }
        }
    }

    // Back substitution: solve UX = Y.
    for (std::size_t ii = n; ii-- > 0;) {
        for (std::size_t k = ii + 1; k < n; ++k) {
            const double u_ik = lu[ii * n + k];

            for (std::size_t j = 0; j < n; ++j) {
                inverse(ii, j) -= u_ik * inverse(k, j);
            }
        }

        const double diagonal = lu[ii * n + ii];

        for (std::size_t j = 0; j < n; ++j) {
            inverse(ii, j) /= diagonal;
        }
    }

    return inverse;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);

    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }

    return result;


}

// solves the linear system Ax = b using LU decomposition with partial pivoting.
Matrix Matrix::solve(const Matrix& b) const {
    if (rows_ != cols_) {
        throw std::invalid_argument("The matrix must be square");
    }

    if (b.rows_ != rows_) {
        throw std::invalid_argument(
            "The number of rows of B must match the size of A"
        );
    }

    const std::size_t n = rows_;
    const std::size_t rhs_cols = b.cols_;

    if (n == 0) {
        return Matrix(0, rhs_cols);
    }

    // Store L and U in a single contiguous buffer:
    // - U is stored on and above the diagonal.
    // - L is stored below the diagonal.
    // - The diagonal of L is implicitly equal to 1.
    std::vector<double> lu = data_;

    // Store the row permutation performed at each elimination step.
    std::vector<std::size_t> pivots(n);

    // Compute the LU factorization with partial pivoting.
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot_row = k;
        double max_value = std::abs(lu[k * n + k]);

        for (std::size_t i = k + 1; i < n; ++i) {
            const double value = std::abs(lu[i * n + k]);

            if (value > max_value) {
                max_value = value;
                pivot_row = i;
            }
        }

        // A missing nonzero pivot means that A is singular.
        if (max_value == 0.0) {
            throw std::runtime_error("The matrix is singular");
        }

        pivots[k] = pivot_row;

        // Move the selected pivot onto the diagonal.
        if (pivot_row != k) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(
                    lu[k * n + j],
                    lu[pivot_row * n + j]
                );
            }
        }

        const double pivot = lu[k * n + k];

        // Compute the multipliers of L and update U.
        for (std::size_t i = k + 1; i < n; ++i) {
            lu[i * n + k] /= pivot;
            const double multiplier = lu[i * n + k];

            for (std::size_t j = k + 1; j < n; ++j) {
                lu[i * n + j] -= multiplier * lu[k * n + j];
            }
        }
    }

    // The solution initially contains a copy of B.
    Matrix solution = b;

    // Apply to B the same row permutations used for A.
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t pivot_row = pivots[k];

        if (pivot_row != k) {
            for (std::size_t j = 0; j < rhs_cols; ++j) {
                std::swap(
                    solution(k, j),
                    solution(pivot_row, j)
                );
            }
        }
    }

    // Forward substitution: solve LY = P * B.
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < i; ++k) {
            const double l_ik = lu[i * n + k];

            for (std::size_t j = 0; j < rhs_cols; ++j) {
                solution(i, j) -= l_ik * solution(k, j);
            }
        }
    }

    // Back substitution: solve UX = Y.
    for (std::size_t ii = n; ii-- > 0;) {
        for (std::size_t k = ii + 1; k < n; ++k) {
            const double u_ik = lu[ii * n + k];

            for (std::size_t j = 0; j < rhs_cols; ++j) {
                solution(ii, j) -= u_ik * solution(k, j);
            }
        }

        const double diagonal = lu[ii * n + ii];

        for (std::size_t j = 0; j < rhs_cols; ++j) {
            solution(ii, j) /= diagonal;
        }
    }

    return solution;
}

Matrix Matrix::solve_cholesky(const Matrix& b) const {
    if (rows_ != cols_) {
        throw std::invalid_argument("The matrix must be square");
    }

    if (b.rows_ != rows_) {
        throw std::invalid_argument(
            "The number of rows of B must match the size of A"
        );
    }

    const std::size_t n = rows_;
    const std::size_t rhs_cols = b.cols_;

    if (n == 0) {
        return Matrix(0, rhs_cols);
    }

    // Compute the lower-triangular Cholesky factor L such that:
    //
    //     A = L * L^T
    Matrix lower(n, n);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double value = (*this)(i, j);

            for (std::size_t k = 0; k < j; ++k) {
                value -= lower(i, k) * lower(j, k);
            }

            if (i == j) {
                if (value <= 0.0) {
                    throw std::runtime_error(
                        "The matrix is not positive definite"
                    );
                }

                lower(i, j) = std::sqrt(value);
            } else {
                lower(i, j) = value / lower(j, j);
            }
        }
    }

    Matrix solution = b;

    // Forward substitution: solve L * Y = B.
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t column = 0; column < rhs_cols; ++column) {
            double value = solution(i, column);

            for (std::size_t k = 0; k < i; ++k) {
                value -= lower(i, k) * solution(k, column);
            }

            solution(i, column) = value / lower(i, i);
        }
    }

    // Back substitution: solve L^T * X = Y.
    for (std::size_t ii = n; ii-- > 0;) {
        for (std::size_t column = 0; column < rhs_cols; ++column) {
            double value = solution(ii, column);

            for (std::size_t k = ii + 1; k < n; ++k) {
                value -= lower(k, ii) * solution(k, column);
            }

            solution(ii, column) = value / lower(ii, ii);
        }
    }

    return solution;
}
} // namespace chiikaml
