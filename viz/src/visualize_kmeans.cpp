#include "chiikaml/viz/visualize_kmeans.hpp"

#include <nlohmann/json.hpp>
#include <stdexcept>

#include "chiikaml/viz/dashboard.hpp"

namespace chiikaml::viz {

void visualize(const KMeans& model, const Matrix& X) {
    if (X.cols() != 2) {
        throw std::invalid_argument("viz::visualize(KMeans) ne supporte que des donnees 2D pour l'instant");
    }

    nlohmann::json payload;
    payload["type"] = "scatter";
    payload["title"] = "KMeans clustering";

    const auto& labels = model.labels();
    payload["points"] = nlohmann::json::array();
    for (std::size_t i = 0; i < X.rows(); ++i) {
        payload["points"].push_back({{"x", X(i, 0)}, {"y", X(i, 1)}, {"cluster", labels[i]}});
    }

    const Matrix& centroids = model.centroids();
    payload["centroids"] = nlohmann::json::array();
    for (std::size_t i = 0; i < centroids.rows(); ++i) {
        payload["centroids"].push_back({{"x", centroids(i, 0)}, {"y", centroids(i, 1)}});
    }

    Dashboard::instance().show(payload.dump());
}

} // namespace chiikaml::viz
