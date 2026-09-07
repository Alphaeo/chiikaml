"""Compare chiikaml aux implementations equivalentes de scikit-learn,
sur les memes donnees : coherence des resultats (les deux
implementations sont-elles d'accord ?) et vitesse (fit/predict).

Pas aussi rigoureux que les benchmarks Google Benchmark (Phase 7) --
ici on utilise juste time.perf_counter() autour d'un seul appel, pas
de warm-up ni de moyenne statistique. Le but ici est de comparer deux
BIBLIOTHEQUES differentes sur le meme travail, pas de mesurer un
micro-gain interne a chiikaml -- la precision de Google Benchmark
n'est pas necessaire pour ca.

Usage (depuis la racine du repo, apres avoir compile les bindings) :
    $env:PYTHONPATH = "build/python"
    python python/benchmarks/compare_sklearn.py
"""

import os

# Doit etre fixe AVANT d'importer numpy/sklearn : sur Windows, le
# KMeans de sklearn (backend MKL/OpenBLAS) a un bug de performance
# documente ou une sursouscription de threads OpenMP le rend ~20x
# plus lent que necessaire (mesure : ~1500ms contre ~78ms avec
# OMP_NUM_THREADS=1, sur les memes donnees). Sans ce correctif, le
# chiffre sklearn releve serait un artefact de plateforme, pas une
# vraie mesure de l'algorithme -- on veut comparer les algorithmes,
# pas un bug de threading Windows.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time

# Meme piege que pour les tests pytest (voir python/tests/conftest.py) :
# depuis Python 3.8, le PATH seul ne suffit plus pour qu'import trouve
# les DLL runtime MinGW dont le module natif chiikaml.pyd depend.
if os.name == "nt":
    _mingw_bin = r"C:\msys64\ucrt64\bin"
    if os.path.isdir(_mingw_bin):
        os.add_dll_directory(_mingw_bin)

import numpy as np
from sklearn.cluster import KMeans as SKKMeans
from sklearn.ensemble import RandomForestClassifier as SKRandomForest
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier as SKDecisionTree
from sklearn.linear_model import LinearRegression

import chiikaml


def timeit(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def make_two_clusters(n_per_cluster=500, seed=42):
    rng = np.random.default_rng(seed)
    cluster0 = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(n_per_cluster, 2))
    cluster1 = rng.normal(loc=[10.0, 10.0], scale=1.0, size=(n_per_cluster, 2))
    X = np.vstack([cluster0, cluster1])
    y = np.array([0] * n_per_cluster + [1] * n_per_cluster)
    return X, y


def to_chiikaml_matrix(X):
    m = chiikaml.Matrix(X.shape[0], X.shape[1])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            m[i, j] = float(X[i, j])
    return m


def report(name, fit_c, predict_c, fit_sk, predict_sk, agreement_label, agreement):
    print(f"=== {name} ===")
    print(f"  chiikaml : fit {fit_c:8.2f} ms   predict {predict_c:8.2f} ms")
    print(f"  sklearn  : fit {fit_sk:8.2f} ms   predict {predict_sk:8.2f} ms")
    print(f"  {agreement_label} : {agreement}")
    print()


def compare_knn(X, y, X_chiikaml, y_list):
    knn_c = chiikaml.KNNClassifier(5)
    _, fit_c = timeit(knn_c.fit, X_chiikaml, y_list)
    preds_c, predict_c = timeit(knn_c.predict, X_chiikaml)

    knn_sk = KNeighborsClassifier(n_neighbors=5)
    _, fit_sk = timeit(knn_sk.fit, X, y)
    preds_sk, predict_sk = timeit(knn_sk.predict, X)

    agreement = np.mean(np.array(preds_c) == preds_sk)
    report("KNNClassifier vs KNeighborsClassifier", fit_c, predict_c, fit_sk, predict_sk,
           "accord des predictions", f"{agreement * 100:.1f}%")


def compare_decision_tree(X, y, X_chiikaml, y_list):
    tree_c = chiikaml.DecisionTreeClassifier(5, 2)
    _, fit_c = timeit(tree_c.fit, X_chiikaml, y_list)
    preds_c, predict_c = timeit(tree_c.predict, X_chiikaml)

    tree_sk = SKDecisionTree(max_depth=5, min_samples_split=2, random_state=42)
    _, fit_sk = timeit(tree_sk.fit, X, y)
    preds_sk, predict_sk = timeit(tree_sk.predict, X)

    agreement = np.mean(np.array(preds_c) == preds_sk)
    report("DecisionTreeClassifier vs DecisionTreeClassifier (sklearn)", fit_c, predict_c, fit_sk,
           predict_sk, "accord des predictions", f"{agreement * 100:.1f}%")


def compare_random_forest(X, y, X_chiikaml, y_list):
    forest_c = chiikaml.RandomForestClassifier(10, 5, 2, 42)
    _, fit_c = timeit(forest_c.fit, X_chiikaml, y_list)
    preds_c, predict_c = timeit(forest_c.predict, X_chiikaml)

    forest_sk = SKRandomForest(n_estimators=10, max_depth=5, min_samples_split=2, random_state=42)
    _, fit_sk = timeit(forest_sk.fit, X, y)
    preds_sk, predict_sk = timeit(forest_sk.predict, X)

    # Le hasard interne (tirage bootstrap, ordre des threads) differe
    # d'une bibliotheque a l'autre -- on compare l'EXACTITUDE de
    # chacune plutot que l'accord exact des predictions, plus juste
    # ici que pour KNN/DecisionTree qui sont deterministes.
    accuracy_c = np.mean(np.array(preds_c) == y)
    accuracy_sk = np.mean(preds_sk == y)
    report("RandomForestClassifier vs RandomForestClassifier (sklearn)", fit_c, predict_c, fit_sk,
           predict_sk, "exactitude (chiikaml vs sklearn)", f"{accuracy_c * 100:.1f}% vs {accuracy_sk * 100:.1f}%")


def compare_kmeans(X, X_chiikaml):
    km_c = chiikaml.KMeans(2, 100, 42)
    _, fit_c = timeit(km_c.fit, X_chiikaml)
    labels_c = km_c.labels()
    predict_c = 0.0

    # n_init=1, pas le defaut sklearn (10) : chiikaml.KMeans ne fait
    # qu'UNE SEULE relance (voir kmeans.hpp -- pas de k-means++, pas
    # de multi-restart pour cette v1). Comparer contre le defaut
    # sklearn (10 relances, qui garde la meilleure) mesurerait 10x
    # plus de travail cote sklearn -- pas le meme algorithme. Avec
    # n_init=10 (defaut reel de sklearn en pratique), sklearn est
    # environ 10x plus lent que ci-dessous mais plus robuste a un
    # mauvais tirage initial -- un vrai compromis, pas juste une
    # question de vitesse brute.
    km_sk = SKKMeans(n_clusters=2, n_init=1, random_state=42)
    _, fit_sk = timeit(km_sk.fit, X)
    labels_sk = km_sk.labels_
    predict_sk = 0.0

    # Le NUMERO de cluster est arbitraire d'une implementation a
    # l'autre (rien ne garantit que "cluster 0" designe le meme
    # groupe des deux cotes) -- l'ARI (Adjusted Rand Index) compare
    # deux partitionnements en ignorant cette permutation. 1.0 =
    # partitionnements identiques, 0.0 = pas mieux qu'au hasard.
    ari = adjusted_rand_score(list(labels_c), list(labels_sk))
    report("KMeans vs KMeans (sklearn)", fit_c, predict_c, fit_sk, predict_sk,
           "accord structurel (Adjusted Rand Index)", f"{ari:.3f}")

def compare_linear_regression(X, y, X_chiikaml, y_list):
    lr_c = chiikaml.LinearRegression(fit_intercept=True)
    _, fit_c = timeit(lr_c.fit, X_chiikaml, y_list)
    preds_c, predict_c = timeit(lr_c.predict, X_chiikaml)

    lr_sk = LinearRegression(fit_intercept=True)
    _, fit_sk = timeit(lr_sk.fit, X, y)
    preds_sk, predict_sk = timeit(lr_sk.predict, X)

    agreement = np.mean(np.isclose(preds_c, preds_sk))
    report("LinearRegression vs LinearRegression (sklearn)", fit_c, predict_c, fit_sk,
           predict_sk, "accord des predictions", f"{agreement * 100:.1f}%")

def main():
    X, y = make_two_clusters(n_per_cluster=500)
    X_chiikaml = to_chiikaml_matrix(X)
    y_list = y.tolist()

    compare_knn(X, y, X_chiikaml, y_list)
    compare_decision_tree(X, y, X_chiikaml, y_list)
    compare_random_forest(X, y, X_chiikaml, y_list)
    compare_kmeans(X, X_chiikaml)
    compare_linear_regression(X, y, X_chiikaml, y_list)


if __name__ == "__main__":
    main()
