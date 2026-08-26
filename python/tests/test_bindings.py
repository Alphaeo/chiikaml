# Tests pytest pour les bindings chiikaml -- le pendant Python de nos
# tests Catch2 : ils decrivent le comportement attendu, ils echouent
# tant que bindings.cpp n'est pas complete.
#
# Pour les lancer, il faut d'abord compiler le module (depuis la
# racine du projet) :
#   cmake -S . -B build -DCHIIKAML_BUILD_PYTHON=ON
#   cmake --build build --target chiikaml_python
# puis lancer pytest avec le dossier de sortie du module sur le
# PYTHONPATH, par exemple (PowerShell) :
#   $env:PYTHONPATH = "build/python"
#   python -m pytest python/tests
# (conftest.py s'occupe automatiquement du chargement des DLL runtime
# MinGW, pas besoin de s'en soucier ici)

import chiikaml


def make_matrix():
    m = chiikaml.Matrix(2, 2)
    m[0, 0] = 1.0
    m[0, 1] = 2.0
    m[1, 0] = 3.0
    m[1, 1] = 4.0
    return m


def test_matrix_dimensions_and_zero_init():
    m = chiikaml.Matrix(3, 4)
    assert m.rows() == 3
    assert m.cols() == 4
    assert m[0, 0] == 0.0


def test_matrix_getitem_setitem():
    m = make_matrix()
    assert m[0, 0] == 1.0
    assert m[1, 1] == 4.0


def test_matrix_repr_is_readable():
    m = make_matrix()
    text = repr(m)
    assert "1" in text
    assert "4" in text


def test_knn_classifier_fit_predict():
    x_train = chiikaml.Matrix(4, 2)
    x_train[0, 0], x_train[0, 1] = 0.0, 0.0
    x_train[1, 0], x_train[1, 1] = 0.0, 1.0
    x_train[2, 0], x_train[2, 1] = 9.0, 9.0
    x_train[3, 0], x_train[3, 1] = 9.0, 8.0
    y_train = [0, 0, 1, 1]

    knn = chiikaml.KNNClassifier(1)
    knn.fit(x_train, y_train)

    query = chiikaml.Matrix(1, 2)
    query[0, 0], query[0, 1] = 8.5, 8.5

    predictions = knn.predict(query)
    assert predictions == [1]


def test_kdtree_nearest_neighbors():
    points = chiikaml.Matrix(3, 2)
    points[0, 0], points[0, 1] = 0.0, 0.0
    points[1, 0], points[1, 1] = 10.0, 10.0
    points[2, 0], points[2, 1] = 10.0, 11.0

    tree = chiikaml.KDTree(points)

    query = chiikaml.Matrix(1, 2)
    query[0, 0], query[0, 1] = 10.0, 10.0

    neighbors = tree.nearest_neighbors(query, 0, 1)
    assert neighbors == [1]


def test_kmeans_fit_predict():
    points = chiikaml.Matrix(4, 2)
    points[0, 0], points[0, 1] = 0.0, 0.0
    points[1, 0], points[1, 1] = 0.0, 1.0
    points[2, 0], points[2, 1] = 10.0, 10.0
    points[3, 0], points[3, 1] = 10.0, 11.0

    km = chiikaml.KMeans(2)
    km.fit(points)

    labels = km.labels()
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
