// Premier benchmark du projet : k-NN brute-force (KNNClassifier)
// contre l'index KD-Tree (KDTree), sur le meme jeu de requetes, a
// plusieurs tailles de dataset. Objectif : verifier avec de vrais
// chiffres si le KD-Tree tient sa promesse ("Phase 3 - KD-Tree pour
// un k-NN rapide"), jamais mesuree jusqu'ici.
//
// Notions Google Benchmark utilisees ici (nouvelles dans ce projet) :
//
// - `static void BM_xxx(benchmark::State& state)` : la signature
//   obligatoire d'une fonction de benchmark. `state` pilote combien
//   de fois la boucle ci-dessous s'execute (Google Benchmark ajuste
//   ca tout seul pour avoir une mesure statistiquement fiable -- tu
//   n'as pas a choisir un nombre d'iterations toi-meme).
//
// - `for (auto _ : state) { ... }` : le coeur mesure. Tout ce qui est
//   AVANT cette boucle (construire le dataset, entrainer le modele)
//   ne compte PAS dans le temps mesure -- seul l'interieur de la
//   boucle est chronometre. C'est crucial : ca permet de benchmarker
//   uniquement le cout d'une PREDICTION, sans inclure le cout de la
//   construction du modele.
//
// - `benchmark::DoNotOptimize(x)` : empeche le compilateur de
//   remarquer "ce resultat n'est jamais utilise" et de supprimer le
//   calcul entier (une optimisation reelle et frequente en -O3 qui
//   fausserait completement un benchmark si on ne s'en protegeait
//   pas).
//
// - `BENCHMARK(BM_xxx)->Arg(N)` : enregistre le benchmark, execute
//   une fois par valeur passee a `state.range(0)` a l'interieur de la
//   fonction -- ici, la taille du dataset d'entrainement. Plusieurs
//   `->Arg(...)` (ou `->Range(min, max)`) permettent de voir comment
//   le temps evolue quand N grandit -- exactement ce qu'il faut pour
//   comparer une complexite O(n) (brute-force) a une complexite
//   moyenne O(log n) (KD-Tree).
//
// - `->Complexity(benchmark::oN)` / `->Complexity(benchmark::oLogN)` :
//   demande a Google Benchmark d'estimer, a partir des mesures a
//   differentes tailles, quelle courbe de complexite s'ajuste le
//   mieux -- une verification empirique de ce qu'on attend
//   theoriquement.

#include <benchmark/benchmark.h>

#include <random>

#include "chiikaml/kdtree.hpp"
#include "chiikaml/knn.hpp"
#include "chiikaml/matrix.hpp"

using namespace chiikaml;

namespace {

// Genere n points aleatoires en `dims` dimensions, seed fixe pour
// que chaque taille de benchmark parte des memes donnees "en style"
// (meme distribution), reproductible d'une execution a l'autre.
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

constexpr std::size_t kDims = 3;
constexpr std::size_t kQueries = 100;
constexpr std::size_t kNeighbors = 5;

} // namespace

static void BM_KNN_BruteForce_Predict(benchmark::State& state) {
    std::size_t n = static_cast<std::size_t>(state.range(0));

    // Tout ce qui suit (jusqu'a la boucle) NE COMPTE PAS dans le
    // temps mesure : c'est la preparation, pas ce qu'on veut chronometrer.
    Matrix X_train = random_points(n, kDims, 42);
    std::vector<int> y_train(n, 0);
    KNNClassifier knn(kNeighbors);
    knn.fit(X_train, y_train);
    Matrix queries = random_points(kQueries, kDims, 123);

    // Indique explicitement a Google Benchmark quelle valeur utiliser
    // comme "N" pour l'ajustement de courbe de complexite -- sans
    // ca, les lignes _BigO/_RMS restent a "nan".
    state.SetComplexityN(static_cast<benchmark::ComplexityN>(n));

    for (auto _ : state) {
        auto predictions = knn.predict(queries);
        benchmark::DoNotOptimize(predictions);
    }
}
BENCHMARK(BM_KNN_BruteForce_Predict)->Range(1 << 7, 1 << 15)->Complexity(benchmark::oN);

static void BM_KDTree_NearestNeighbors_Predict(benchmark::State& state) {
    std::size_t n = static_cast<std::size_t>(state.range(0));

    Matrix X_train = random_points(n, kDims, 42);
    KDTree tree(X_train);
    Matrix queries = random_points(kQueries, kDims, 123);

    state.SetComplexityN(static_cast<benchmark::ComplexityN>(n));

    for (auto _ : state) {
        for (std::size_t i = 0; i < queries.rows(); ++i) {
            auto neighbors = tree.nearest_neighbors(queries, i, kNeighbors);
            benchmark::DoNotOptimize(neighbors);
        }
    }
}
BENCHMARK(BM_KDTree_NearestNeighbors_Predict)->Range(1 << 7, 1 << 15)->Complexity(benchmark::oLogN);

BENCHMARK_MAIN();
