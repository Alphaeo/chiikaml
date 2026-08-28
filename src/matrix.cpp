#include "chiikaml/matrix.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
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

} // namespace chiikaml
