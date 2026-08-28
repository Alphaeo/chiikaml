// Troisieme benchmark : verifier empiriquement le pari fait des la
// Phase 1 (voir le commentaire de tete de matrix.hpp) -- un seul
// bloc contigu (Matrix) devrait etre plus rapide a parcourir qu'un
// vector<vector<double>> (une ligne = une allocation separee,
// dispersee dans le tas).
//
// L'operation elle-meme (sommer tous les elements) est volontairement
// triviale et identique des deux cotes -- toute la difference de
// temps ne peut venir que du layout memoire, pas de l'arithmetique.

#include <benchmark/benchmark.h>

#include <vector>

#include "chiikaml/matrix.hpp"

using namespace chiikaml;

namespace {

// Repere de comparaison volontairement naif, avec la meme interface
// que Matrix pour pouvoir reutiliser le meme sum_all() sur les deux.
class NaiveMatrix {
public:
    NaiveMatrix(std::size_t rows, std::size_t cols) : data_(rows, std::vector<double>(cols, 0.0)) {}

    double& operator()(std::size_t row, std::size_t col) { return data_[row][col]; }
    double operator()(std::size_t row, std::size_t col) const { return data_[row][col]; }
    std::size_t rows() const { return data_.size(); }
    std::size_t cols() const { return data_.empty() ? 0 : data_[0].size(); }

private:
    std::vector<std::vector<double>> data_;
};

template <typename MatrixLike>
double sum_all(const MatrixLike& m) {
    double total = 0.0;
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j) {
            total += m(i, j);
        }
    }
    return total;
}

} // namespace

static void BM_Matrix_FullTraversal(benchmark::State& state) {
    std::size_t n = static_cast<std::size_t>(state.range(0));
    Matrix m(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            m(i, j) = static_cast<double>(i + j);
        }
    }

    state.SetComplexityN(static_cast<benchmark::ComplexityN>(n * n));
    for (auto _ : state) {
        double total = sum_all(m);
        benchmark::DoNotOptimize(total);
    }
}
BENCHMARK(BM_Matrix_FullTraversal)->Arg(64)->Arg(256)->Arg(1024)->Arg(2048)->Complexity(benchmark::oN);

static void BM_NaiveMatrix_FullTraversal(benchmark::State& state) {
    std::size_t n = static_cast<std::size_t>(state.range(0));
    NaiveMatrix m(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            m(i, j) = static_cast<double>(i + j);
        }
    }

    state.SetComplexityN(static_cast<benchmark::ComplexityN>(n * n));
    for (auto _ : state) {
        double total = sum_all(m);
        benchmark::DoNotOptimize(total);
    }
}
BENCHMARK(BM_NaiveMatrix_FullTraversal)->Arg(64)->Arg(256)->Arg(1024)->Arg(2048)->Complexity(benchmark::oN);

// Deuxieme forme, plus proche de ce que chiikaml manipule REELLEMENT
// (un jeu d'entrainement : beaucoup de lignes/points, peu de
// colonnes/features) plutot qu'une matrice carree. Hypothese : avec
// des lignes courtes, le cout "un saut de pointeur par ligne" de
// NaiveMatrix a moins de travail utile (contigu) sur lequel
// s'amortir -- l'ecart devrait etre plus marque ici qu'en carre.
static void BM_Matrix_TallThinTraversal(benchmark::State& state) {
    std::size_t n_rows = static_cast<std::size_t>(state.range(0));
    constexpr std::size_t kCols = 4;
    Matrix m(n_rows, kCols);
    for (std::size_t i = 0; i < n_rows; ++i) {
        for (std::size_t j = 0; j < kCols; ++j) {
            m(i, j) = static_cast<double>(i + j);
        }
    }

    state.SetComplexityN(static_cast<benchmark::ComplexityN>(n_rows));
    for (auto _ : state) {
        double total = sum_all(m);
        benchmark::DoNotOptimize(total);
    }
}
BENCHMARK(BM_Matrix_TallThinTraversal)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->Complexity(benchmark::oN);

static void BM_NaiveMatrix_TallThinTraversal(benchmark::State& state) {
    std::size_t n_rows = static_cast<std::size_t>(state.range(0));
    constexpr std::size_t kCols = 4;
    NaiveMatrix m(n_rows, kCols);
    for (std::size_t i = 0; i < n_rows; ++i) {
        for (std::size_t j = 0; j < kCols; ++j) {
            m(i, j) = static_cast<double>(i + j);
        }
    }

    state.SetComplexityN(static_cast<benchmark::ComplexityN>(n_rows));
    for (auto _ : state) {
        double total = sum_all(m);
        benchmark::DoNotOptimize(total);
    }
}
BENCHMARK(BM_NaiveMatrix_TallThinTraversal)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->Complexity(benchmark::oN);
