#pragma once

#include <vector>

#include "chiikaml/matrix.hpp"


namespace chiikaml::metrics {
// Regression metrics for evaluating model performance.


double mean_squared_error(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred
);

double mean_absolute_error(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred
);

double r2_score(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred
);


}