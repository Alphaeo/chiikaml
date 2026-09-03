#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include "chiikaml/decision_tree.hpp"
#include "chiikaml/kdtree.hpp"
#include "chiikaml/kmeans.hpp"
#include "chiikaml/knn.hpp"
#include "chiikaml/matrix.hpp"
#include "chiikaml/random_forest.hpp"
#include "chiikaml/linear_regression.hpp"

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

    py::class_<KNNClassifier>(m, "KNNClassifier")
        .def(py::init<std::size_t>(), py::arg("k"))
        .def("fit", &KNNClassifier::fit, py::arg("X"), py::arg("y"))
        .def("predict", &KNNClassifier::predict, py::arg("X"));

    py::class_<KDTree>(m, "KDTree")
        .def(py::init<const Matrix&>(), py::arg("points"))
        .def("nearest_neighbors", &KDTree::nearest_neighbors, py::arg("query"), py::arg("query_row"),
             py::arg("k"));

    py::class_<KMeans>(m, "KMeans")
        .def(py::init<std::size_t, std::size_t, unsigned int>(), py::arg("n_clusters"),
             py::arg("max_iterations") = 100, py::arg("seed") = 42)
        .def("fit", &KMeans::fit, py::arg("X"))
        .def("predict", &KMeans::predict, py::arg("X"))
        .def("labels", &KMeans::labels, py::return_value_policy::copy)
        .def("centroids", &KMeans::centroids, py::return_value_policy::copy);

    py::class_<DecisionTreeClassifier>(m, "DecisionTreeClassifier")
        .def(py::init<std::size_t, std::size_t>(), py::arg("max_depth") = 5, py::arg("min_samples_split") = 2)
        .def("fit", &DecisionTreeClassifier::fit, py::arg("X"), py::arg("y"))
        .def("predict", &DecisionTreeClassifier::predict, py::arg("X"));

    py::class_<RandomForestClassifier>(m, "RandomForestClassifier")
        .def(py::init<std::size_t, std::size_t, std::size_t, unsigned int>(), py::arg("n_trees") = 10,
             py::arg("max_depth") = 5, py::arg("min_samples_split") = 2, py::arg("seed") = 42)
        .def("fit", &RandomForestClassifier::fit, py::arg("X"), py::arg("y"))
        .def("predict", &RandomForestClassifier::predict, py::arg("X"));

    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<bool>(), py::arg("fit_intercept") = true)
        .def("fit", &LinearRegression::fit, py::arg("X"), py::arg("y"))
        .def("predict", &LinearRegression::predict, py::arg("X"))
        .def("coefficients", &LinearRegression::coefficients, py::return_value_policy::copy)
        .def("intercept", &LinearRegression::intercept);

}
