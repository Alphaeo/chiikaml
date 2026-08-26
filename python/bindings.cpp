#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include "chiikaml/kdtree.hpp"
#include "chiikaml/kmeans.hpp"
#include "chiikaml/knn.hpp"
#include "chiikaml/matrix.hpp"

namespace py = pybind11;
using namespace chiikaml;

// pybind11/stl.h : convertit automatiquement std::vector<int>,
// std::vector<std::size_t>, etc. en listes Python et vice-versa --
// c'est ce qui permet a KNNClassifier::predict() (qui renvoie un
// std::vector<int> cote C++) de devenir directement une liste Python
// sans code de conversion ecrit a la main.

PYBIND11_MODULE(chiikaml, m) {
    m.doc() = "Bindings Python pour chiikaml";

    py::class_<Matrix>(m, "Matrix")
        .def(py::init<std::size_t, std::size_t>())
        .def("rows", &Matrix::rows)
        .def("cols", &Matrix::cols)
        .def("__getitem__",
             [](const Matrix& self, std::pair<std::size_t, std::size_t> idx) {
                 return self(idx.first, idx.second);
             })
        .def("__setitem__",
             [](Matrix& self, std::pair<std::size_t, std::size_t> idx, double value) {
                 self(idx.first, idx.second) = value;
             })
        .def("__repr__",
             [](const Matrix& self) {
                 std::ostringstream oss;
                 oss << self;
                 return oss.str();
             })
        .def(py::self + py::self);

    // TODO(toi) : bind KNNClassifier, KDTree, KMeans.
    // Une fois Matrix fait, ces trois-la suivent le meme schema simple :
    // constructeur + methodes publiques bindees une par une avec
    // `.def("nom_python", &Classe::methode)`. Aucune n'a besoin
    // d'adaptation particuliere comme Matrix (pas d'operator(),
    // pas d'operator<<) -- leurs methodes publiques (fit, predict,
    // nearest_neighbors, labels, centroids...) se bindent telles
    // quelles.
}
