# chiikaml

Bibliotheque C++ d'algorithmes de machine learning "classiques",
implementes depuis zero avec un vrai focus sur la performance
(structures de donnees cache-friendly, multithreading, SIMD).

Projet pedagogique : chaque module est construit etape par etape,
avec des tests comme specification.

## Roadmap

- [ ] Phase 0 - Setup (CMake, Catch2, structure du projet)
- [ ] Phase 1 - `Matrix` : stockage contigu, operations de base
- [ ] Phase 2 - k-NN (brute-force) + metriques de distance
- [ ] Phase 3 - KD-Tree pour un k-NN rapide
- [ ] Phase 4 - K-Means
- [ ] Phase 5 - Arbre de decision (CART, gini/entropie)
- [ ] Phase 6 - Random Forest (bagging + multithreading)
- [ ] Phase 7 - Optimisation : benchmarks, profiling, SIMD, cache
- [ ] Phase 8 - Polish : CLI, benchmarks vs sklearn, CI

## Build

```
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build
```
