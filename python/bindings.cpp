#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "chiikaml/binary_svm.hpp"
#include "chiikaml/decision_tree.hpp"
#include "chiikaml/kdtree.hpp"
#include "chiikaml/kmeans.hpp"
#include "chiikaml/knn.hpp"
#include "chiikaml/linear_regression.hpp"
#include "chiikaml/matrix.hpp"
#include "chiikaml/metrics/classification_metrics.hpp"
#include "chiikaml/model_selection/grid_search.hpp"
#include "chiikaml/model_selection/k_fold.hpp"
#include "chiikaml/model_selection/train_test_split.hpp"
#include "chiikaml/random_forest.hpp"
#include "chiikaml/svm_kernels.hpp"

namespace py = pybind11;

using namespace chiikaml;

namespace {

struct KNNGridParameters {
    std::size_t k;
};

struct BinarySVMGridParameters {
    double C;
    SVMKernel kernel;
    double gamma;
};

model_selection::ClassificationMetric
get_classification_metric(const std::string& name) {
    if (name == "accuracy") {
        return metrics::accuracy;
    }

    if (name == "precision") {
        return metrics::precision;
    }

    if (name == "recall") {
        return metrics::recall;
    }

    if (name == "f1" || name == "f1_score") {
        return metrics::f1_score;
    }

    throw std::invalid_argument(
        "Unknown classification metric: " + name
    );
}

} // namespace

PYBIND11_MODULE(chiikaml, m) {
    m.doc() = "Python bindings for chiikaml";

    py::class_<Matrix>(m, "Matrix")
        .def(
            py::init<std::size_t, std::size_t>(),
            py::arg("rows"),
            py::arg("cols")
        )
        .def(
            "rows",
            &Matrix::rows
        )
        .def(
            "cols",
            &Matrix::cols
        )
        .def(
            "select_rows",
            &Matrix::select_rows,
            py::arg("indices")
        )
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

    py::enum_<SVMKernel>(m, "SVMKernel")
        .value("Linear", SVMKernel::Linear)
        .value("Polynomial", SVMKernel::Polynomial)
        .value("RBF", SVMKernel::RBF);

    py::class_<BinarySVM>(m, "BinarySVM")
        .def(
            py::init<
                double,
                SVMKernel,
                double,
                std::size_t,
                double,
                std::size_t,
                double,
                bool,
                unsigned int
            >(),
            py::arg("C") = 1.0,
            py::arg("kernel") = SVMKernel::RBF,
            py::arg("gamma") = 0.0,
            py::arg("degree") = 3,
            py::arg("coef0") = 0.0,
            py::arg("max_iterations") = 1000,
            py::arg("tolerance") = 1e-4,
            py::arg("fit_intercept") = true,
            py::arg("seed") = 42
        )
        .def(
            "fit",
            &BinarySVM::fit,
            py::arg("X"),
            py::arg("y")
        )
        .def(
            "predict",
            &BinarySVM::predict,
            py::arg("X")
        )
        .def(
            "decision_function",
            &BinarySVM::decision_function,
            py::arg("X")
        )
        .def(
            "decision_value",
            &BinarySVM::decision_value,
            py::arg("X"),
            py::arg("sample_row")
        )
        .def(
            "support_vectors",
            &BinarySVM::support_vectors,
            py::return_value_policy::copy
        )
        .def(
            "dual_coefficients",
            &BinarySVM::dual_coefficients,
            py::return_value_policy::copy
        )
        .def(
            "intercept",
            &BinarySVM::intercept
        )
        .def(
            "C",
            &BinarySVM::C
        )
        .def(
            "kernel",
            &BinarySVM::kernel
        )
        .def(
            "gamma",
            &BinarySVM::gamma
        )
        .def(
            "degree",
            &BinarySVM::degree
        )
        .def(
            "coef0",
            &BinarySVM::coef0
        )
        .def(
            "converged",
            &BinarySVM::converged
        )
        .def(
            "iterations",
            &BinarySVM::iterations
        )
        .def(
            "number_of_support_vectors",
            &BinarySVM::number_of_support_vectors
        );

    // Classification metrics
    py::module_ metrics_module = m.def_submodule(
        "metrics",
        "Classification metrics"
    );

    metrics_module.def(
        "accuracy",
        &metrics::accuracy,
        py::arg("y_true"),
        py::arg("y_pred")
    );

    metrics_module.def(
        "precision",
        &metrics::precision,
        py::arg("y_true"),
        py::arg("y_pred")
    );

    metrics_module.def(
        "recall",
        &metrics::recall,
        py::arg("y_true"),
        py::arg("y_pred")
    );

    metrics_module.def(
        "f1_score",
        &metrics::f1_score,
        py::arg("y_true"),
        py::arg("y_pred")
    );

    metrics_module.def(
        "confusion_matrix",
        &metrics::confusion_matrix,
        py::arg("y_true"),
        py::arg("y_pred")
    );

    // Model-selection utilities
    py::module_ model_selection_module = m.def_submodule(
        "model_selection",
        "Dataset splitting and model-selection utilities"
    );

    using ClassificationSplit =
        model_selection::TrainTestSplitResult<int>;

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
            return model_selection::train_test_split<int>(
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

    py::class_<model_selection::KFold>(
        model_selection_module,
        "KFold"
    )
        .def(
            py::init<std::size_t, bool, unsigned int>(),
            py::arg("n_splits") = 5,
            py::arg("shuffle") = false,
            py::arg("seed") = 42
        )
        .def(
            "split",
            &model_selection::KFold::split,
            py::arg("n_samples")
        )
        .def(
            "n_splits",
            &model_selection::KFold::n_splits
        )
        .def(
            "shuffle",
            &model_selection::KFold::shuffle
        )
        .def(
            "seed",
            &model_selection::KFold::seed
        );

    model_selection_module.def(
        "grid_search_knn",
        [](
            const Matrix& X,
            const std::vector<int>& y,
            const std::vector<std::size_t>& k_values,
            std::size_t n_splits,
            bool shuffle,
            unsigned int seed,
            const std::string& metric_name
        ) {
            std::vector<KNNGridParameters> parameter_grid;
            parameter_grid.reserve(k_values.size());

            for (std::size_t k : k_values) {
                parameter_grid.push_back(
                    KNNGridParameters{k}
                );
            }

            const model_selection::KFold splitter(
                n_splits,
                shuffle,
                seed
            );

            const auto metric =
                get_classification_metric(metric_name);

            auto result =
                model_selection::grid_search_classification(
                    parameter_grid,
                    X,
                    y,
                    splitter,
                    [](const KNNGridParameters& parameters) {
                        return KNNClassifier(parameters.k);
                    },
                    metric
                );

            py::list candidates;

            for (const auto& candidate :
                 result.candidate_results) {
                py::dict candidate_result;

                candidate_result["k"] =
                    candidate.parameters.k;

                candidate_result["mean_score"] =
                    candidate.cross_validation.mean_score;

                candidate_result["standard_deviation"] =
                    candidate.cross_validation.standard_deviation;

                candidate_result["fold_scores"] =
                    candidate.cross_validation.fold_scores;

                candidates.append(candidate_result);
            }

            py::dict output;

            output["best_k"] =
                result.best_parameters.k;

            output["best_index"] =
                result.best_index;

            output["best_score"] =
                result.best_score;

            output["candidate_results"] =
                std::move(candidates);

            output["best_model"] =
                py::cast(std::move(result.best_model));

            return output;
        },
        py::arg("X"),
        py::arg("y"),
        py::arg("k_values"),
        py::arg("n_splits") = 5,
        py::arg("shuffle") = true,
        py::arg("seed") = 42,
        py::arg("metric") = "accuracy"
    );

    model_selection_module.def(
        "grid_search_binary_svm",
        [](
            const Matrix& X,
            const std::vector<int>& y,
            const std::vector<double>& C_values,
            const std::vector<SVMKernel>& kernels,
            const std::vector<double>& gamma_values,
            std::size_t n_splits,
            bool shuffle,
            unsigned int seed,
            const std::string& metric_name,
            std::size_t max_iterations,
            double tolerance
        ) {
            std::vector<BinarySVMGridParameters> parameter_grid;

            parameter_grid.reserve(
                C_values.size()
                * kernels.size()
                * gamma_values.size()
            );

            for (double C : C_values) {
                for (SVMKernel kernel : kernels) {
                    for (double gamma : gamma_values) {
                        parameter_grid.push_back(
                            BinarySVMGridParameters{
                                C,
                                kernel,
                                gamma
                            }
                        );
                    }
                }
            }

            const model_selection::KFold splitter(
                n_splits,
                shuffle,
                seed
            );

            const auto metric =
                get_classification_metric(metric_name);

            auto result =
                model_selection::grid_search_classification(
                    parameter_grid,
                    X,
                    y,
                    splitter,
                    [
                        max_iterations,
                        tolerance,
                        seed
                    ](
                        const BinarySVMGridParameters& parameters
                    ) {
                        return BinarySVM(
                            parameters.C,
                            parameters.kernel,
                            parameters.gamma,
                            3,
                            0.0,
                            max_iterations,
                            tolerance,
                            true,
                            seed
                        );
                    },
                    metric
                );

            py::list candidates;

            for (const auto& candidate :
                 result.candidate_results) {
                py::dict candidate_result;

                candidate_result["C"] =
                    candidate.parameters.C;

                candidate_result["kernel"] =
                    candidate.parameters.kernel;

                candidate_result["gamma"] =
                    candidate.parameters.gamma;

                candidate_result["mean_score"] =
                    candidate.cross_validation.mean_score;

                candidate_result["standard_deviation"] =
                    candidate.cross_validation.standard_deviation;

                candidate_result["fold_scores"] =
                    candidate.cross_validation.fold_scores;

                candidates.append(candidate_result);
            }

            py::dict output;

            output["best_C"] =
                result.best_parameters.C;

            output["best_kernel"] =
                result.best_parameters.kernel;

            output["best_gamma"] =
                result.best_parameters.gamma;

            output["best_index"] =
                result.best_index;

            output["best_score"] =
                result.best_score;

            output["candidate_results"] =
                std::move(candidates);

            output["best_model"] =
                py::cast(std::move(result.best_model));

            return output;
        },
        py::arg("X"),
        py::arg("y"),
        py::arg("C_values"),
        py::arg("kernels"),
        py::arg("gamma_values"),
        py::arg("n_splits") = 5,
        py::arg("shuffle") = true,
        py::arg("seed") = 42,
        py::arg("metric") = "accuracy",
        py::arg("max_iterations") = 5000,
        py::arg("tolerance") = 1e-4
    );
}