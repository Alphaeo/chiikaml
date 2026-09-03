#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

#include "chiikaml/decision_tree.hpp"
#include "chiikaml/kdtree.hpp"
#include "chiikaml/kmeans.hpp"
#include "chiikaml/knn.hpp"
#include "chiikaml/linear_regression.hpp"
#include "chiikaml/matrix.hpp"
#include "chiikaml/metrics/classification_metrics.hpp"
#include "chiikaml/model_selection/train_test_split.hpp"
#include "chiikaml/random_forest.hpp"

namespace py = pybind11;

using namespace chiikaml;

PYBIND11_MODULE(chiikaml, m) {
    m.doc() = "Bindings Python pour chiikaml";

    py::class_<Matrix>(m, "Matrix")
        .def(py::init<std::size_t, std::size_t>())
        .def("rows", &Matrix::rows)
        .def("cols", &Matrix::cols)
        .def(
            "__getitem__",
            [](
                const Matrix& self,
                std::pair<std::size_t, std::size_t> index
            ) {
                return self(index.first, index.second);
            }
        )
        .def(
            "__setitem__",
            [](
                Matrix& self,
                std::pair<std::size_t, std::size_t> index,
                double value
            ) {
                self(index.first, index.second) = value;
            }
        )
        .def(
            "__repr__",
            [](const Matrix& self) {
                std::ostringstream output;
                output << self;
                return output.str();
            }
        )
        .def(py::self + py::self);

    py::class_<KNNClassifier>(m, "KNNClassifier")
        .def(
            py::init<std::size_t>(),
            py::arg("k")
        )
        .def(
            "fit",
            &KNNClassifier::fit,
            py::arg("X"),
            py::arg("y")
        )
        .def(
            "predict",
            &KNNClassifier::predict,
            py::arg("X")
        );

    py::class_<KDTree>(m, "KDTree")
        .def(
            py::init<const Matrix&>(),
            py::arg("points")
        )
        .def(
            "nearest_neighbors",
            &KDTree::nearest_neighbors,
            py::arg("query"),
            py::arg("query_row"),
            py::arg("k")
        );

    py::class_<KMeans>(m, "KMeans")
        .def(
            py::init<
                std::size_t,
                std::size_t,
                unsigned int
            >(),
            py::arg("n_clusters"),
            py::arg("max_iterations") = 100,
            py::arg("seed") = 42
        )
        .def(
            "fit",
            &KMeans::fit,
            py::arg("X")
        )
        .def(
            "predict",
            &KMeans::predict,
            py::arg("X")
        )
        .def(
            "labels",
            &KMeans::labels,
            py::return_value_policy::copy
        )
        .def(
            "centroids",
            &KMeans::centroids,
            py::return_value_policy::copy
        );

    py::class_<DecisionTreeClassifier>(
        m,
        "DecisionTreeClassifier"
    )
        .def(
            py::init<std::size_t, std::size_t>(),
            py::arg("max_depth") = 5,
            py::arg("min_samples_split") = 2
        )
        .def(
            "fit",
            &DecisionTreeClassifier::fit,
            py::arg("X"),
            py::arg("y")
        )
        .def(
            "predict",
            &DecisionTreeClassifier::predict,
            py::arg("X")
        );

    py::class_<RandomForestClassifier>(
        m,
        "RandomForestClassifier"
    )
        .def(
            py::init<
                std::size_t,
                std::size_t,
                std::size_t,
                unsigned int
            >(),
            py::arg("n_trees") = 10,
            py::arg("max_depth") = 5,
            py::arg("min_samples_split") = 2,
            py::arg("seed") = 42
        )
        .def(
            "fit",
            &RandomForestClassifier::fit,
            py::arg("X"),
            py::arg("y")
        )
        .def(
            "predict",
            &RandomForestClassifier::predict,
            py::arg("X")
        );

    py::class_<LinearRegression>(
        m,
        "LinearRegression"
    )
        .def(
            py::init<bool>(),
            py::arg("fit_intercept") = true
        )
        .def(
            "fit",
            &LinearRegression::fit,
            py::arg("X"),
            py::arg("y")
        )
        .def(
            "predict",
            &LinearRegression::predict,
            py::arg("X")
        )
        .def(
            "coefficients",
            &LinearRegression::coefficients,
            py::return_value_policy::copy
        )
        .def(
            "intercept",
            &LinearRegression::intercept
        );

    // Classification metrics
    py::module_ metrics_module = m.def_submodule(
        "metrics",
        "Classification metrics"
    );

    metrics_module.def(
        "accuracy",
        &chiikaml::metrics::accuracy,
        py::arg("y_true"),
        py::arg("y_pred")
    );

    metrics_module.def(
        "precision",
        &chiikaml::metrics::precision,
        py::arg("y_true"),
        py::arg("y_pred")
    );

    metrics_module.def(
        "recall",
        &chiikaml::metrics::recall,
        py::arg("y_true"),
        py::arg("y_pred")
    );

    metrics_module.def(
        "f1_score",
        &chiikaml::metrics::f1_score,
        py::arg("y_true"),
        py::arg("y_pred")
    );

    metrics_module.def(
        "confusion_matrix",
        &chiikaml::metrics::confusion_matrix,
        py::arg("y_true"),
        py::arg("y_pred")
    );

    // Train/test split for classification targets.
    using ClassificationSplit =
        chiikaml::model_selection::TrainTestSplitResult<int>;

    py::module_ model_selection_module = m.def_submodule(
        "model_selection",
        "Dataset splitting utilities"
    );

    py::class_<ClassificationSplit>(
        model_selection_module,
        "TrainTestSplitResult"
    )
        .def_readonly(
            "X_train",
            &ClassificationSplit::X_train
        )
        .def_readonly(
            "X_test",
            &ClassificationSplit::X_test
        )
        .def_readonly(
            "y_train",
            &ClassificationSplit::y_train
        )
        .def_readonly(
            "y_test",
            &ClassificationSplit::y_test
        );

    model_selection_module.def(
        "train_test_split",
        [](
            const Matrix& X,
            const std::vector<int>& y,
            double train_size,
            bool shuffle,
            unsigned int seed
        ) {
            return chiikaml::model_selection::
                train_test_split<int>(
                    X,
                    y,
                    train_size,
                    shuffle,
                    seed
                );
        },
        py::arg("X"),
        py::arg("y"),
        py::arg("train_size") = 0.8,
        py::arg("shuffle") = true,
        py::arg("seed") = 42
    );
}