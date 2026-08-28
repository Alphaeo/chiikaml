// Deuxieme benchmark : mesurer le vrai gain du multithreading dans
// RandomForestClassifier::fit(). Probleme : RandomForestClassifier
// ne propose aucun mode "sequentiel" pour comparer -- elle lance
// toujours un thread par arbre. On construit donc ici un
// entrainement sequentiel EQUIVALENT (meme nombre d'arbres, meme
// bootstrap avec remise, meme seed par arbre), pour comparer un
// travail identique avec et sans parallelisme.
//
// Pas de BENCHMARK_MAIN() dans ce fichier : un seul point d'entree
// suffit pour tout l'executable chiikaml_benchmarks, deja fourni par
// bench_knn_vs_kdtree.cpp. Les benchmarks de plusieurs fichiers .cpp
// se retrouvent tous enregistres dans le meme executable.

#include <benchmark/benchmark.h>

#include <random>

#include "chiikaml/decision_tree.hpp"
#include "chiikaml/matrix.hpp"
#include "chiikaml/random_forest.hpp"

using namespace chiikaml;

namespace {

constexpr std::size_t kDims = 3;
constexpr std::size_t kDatasetSize = 4096;
constexpr std::size_t kMaxDepth = 5;
constexpr std::size_t kMinSamplesSplit = 2;

Matrix random_points(std::size_t n, std::size_t dims, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    Matrix X(n, dims);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < dims; ++j) {
            X(i, j) = dist(rng);
        }
    }
    return X;
}

// Labels non triviaux (pas tous identiques) pour que les arbres
// aient un vrai travail de decoupe a faire, pas juste creer une
// feuille des le premier appel.
std::vector<int> threshold_labels(const Matrix& X) {
    std::vector<int> y(X.rows());
    for (std::size_t i = 0; i < X.rows(); ++i) {
        y[i] = X(i, 0) > 0.0 ? 1 : 0;
    }
    return y;
}

// Duplique volontairement RandomForestClassifier::bootstrap_sample
// (privee, donc pas accessible depuis ce fichier) : X.rows() indices
// tires AVEC remise, meme logique.
std::pair<Matrix, std::vector<int>> bootstrap_sample(const Matrix& X, const std::vector<int>& y,
                                                       std::mt19937& rng) {
    Matrix X_sample(X.rows(), X.cols());
    std::vector<int> y_sample;
    y_sample.reserve(X.rows());

    std::uniform_int_distribution<std::size_t> dist(0, X.rows() - 1);
    for (std::size_t i = 0; i < X.rows(); ++i) {
        std::size_t idx = dist(rng);
        for (std::size_t j = 0; j < X.cols(); ++j) {
            X_sample(i, j) = X(idx, j);
        }
        y_sample.push_back(y[idx]);
    }
    return {X_sample, y_sample};
}

} // namespace

static void BM_RandomForest_Fit_Sequential(benchmark::State& state) {
    std::size_t n_trees = static_cast<std::size_t>(state.range(0));
    Matrix X = random_points(kDatasetSize, kDims, 42);
    std::vector<int> y = threshold_labels(X);

    state.SetComplexityN(static_cast<benchmark::ComplexityN>(n_trees));

    for (auto _ : state) {
        std::vector<DecisionTreeClassifier> trees;
        trees.reserve(n_trees);
        for (std::size_t t = 0; t < n_trees; ++t) {
            trees.emplace_back(kMaxDepth, kMinSamplesSplit);
            std::mt19937 rng(42u + static_cast<unsigned int>(t));
            auto [X_sample, y_sample] = bootstrap_sample(X, y, rng);
            trees.back().fit(X_sample, y_sample);
        }
        benchmark::DoNotOptimize(trees);
    }
}
BENCHMARK(BM_RandomForest_Fit_Sequential)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Unit(benchmark::kMillisecond);

static void BM_RandomForest_Fit_Parallel(benchmark::State& state) {
    std::size_t n_trees = static_cast<std::size_t>(state.range(0));
    Matrix X = random_points(kDatasetSize, kDims, 42);
    std::vector<int> y = threshold_labels(X);

    state.SetComplexityN(static_cast<benchmark::ComplexityN>(n_trees));

    for (auto _ : state) {
        RandomForestClassifier forest(n_trees, kMaxDepth, kMinSamplesSplit, 42);
        forest.fit(X, y);
        benchmark::DoNotOptimize(forest);
    }
}
BENCHMARK(BM_RandomForest_Fit_Parallel)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Unit(benchmark::kMillisecond);
