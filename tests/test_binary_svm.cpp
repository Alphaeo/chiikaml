#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

#include "chiikaml/binary_svm.hpp"
#include "chiikaml/matrix.hpp"
#include "chiikaml/svm_kernels.hpp"

using chiikaml::Matrix;
using chiikaml::SVMKernel;
using chiikaml::BinarySVM;

TEST_CASE("BinarySVM learns correctly a linearly separable problem",
          "[svm][binary]") {
    Matrix X(4, 1);

    X(0, 0) = -2.0;
    X(1, 0) = -1.0;
    X(2, 0) =  1.0;
    X(3, 0) =  2.0;

    const std::vector<int> y = {
        -1, -1, 1, 1
    };

    BinarySVM model(
        10.0,
        SVMKernel::Linear,
        0.0,
        3,
        0.0,
        5000,
        1e-6,
        true,
        42
    );

    model.fit(X, y);

    const std::vector<int> predictions =
        model.predict(X);

    REQUIRE(predictions == y);
    REQUIRE(model.kernel() == SVMKernel::Linear);
    REQUIRE(model.C() == 10.0);
    REQUIRE(model.iterations() > 0);
}

TEST_CASE("BinarySVM decision scores have the expected signs",
          "[svm][binary]") {
    Matrix X_train(4, 1);

    X_train(0, 0) = -2.0;
    X_train(1, 0) = -1.0;
    X_train(2, 0) =  1.0;
    X_train(3, 0) =  2.0;

    const std::vector<int> y = {
        -1, -1, 1, 1
    };

    BinarySVM model(
        10.0,
        SVMKernel::Linear
    );

    model.fit(X_train, y);

    Matrix X_test(2, 1);

    X_test(0, 0) = -3.0;
    X_test(1, 0) =  3.0;

    const std::vector<double> scores =
        model.decision_function(X_test);

    REQUIRE(scores.size() == 2);
    REQUIRE(scores[0] < 0.0);
    REQUIRE(scores[1] >= 0.0);

    const std::vector<int> predictions =
        model.predict(X_test);

    REQUIRE(predictions[0] == -1);
    REQUIRE(predictions[1] == 1);
}

TEST_CASE("BinarySVM retains its support vectors",
          "[svm][binary]") {
    Matrix X(4, 1);

    X(0, 0) = -2.0;
    X(1, 0) = -1.0;
    X(2, 0) =  1.0;
    X(3, 0) =  2.0;

    const std::vector<int> y = {
        -1, -1, 1, 1
    };

    BinarySVM model(
        10.0,
        SVMKernel::Linear
    );

    model.fit(X, y);

    const Matrix& support_vectors =
        model.support_vectors();

    const std::vector<double>& dual_coefficients =
        model.dual_coefficients();

    REQUIRE(model.number_of_support_vectors() > 0);
    REQUIRE(model.number_of_support_vectors() <= X.rows());

    REQUIRE(
        support_vectors.rows()
        == model.number_of_support_vectors()
    );

    REQUIRE(support_vectors.cols() == X.cols());

    REQUIRE(
        dual_coefficients.size()
        == model.number_of_support_vectors()
    );

    for (double coefficient : dual_coefficients) {
        REQUIRE(coefficient != 0.0);
    }
}

TEST_CASE("BinarySVM with an RBF kernel learns XOR",
          "[svm][binary][rbf]") {
    Matrix X(4, 2);

    X(0, 0) = 0.0; X(0, 1) = 0.0;
    X(1, 0) = 0.0; X(1, 1) = 1.0;
    X(2, 0) = 1.0; X(2, 1) = 0.0;
    X(3, 0) = 1.0; X(3, 1) = 1.0;

    // XOR is not linearly separable.
    const std::vector<int> y = {
        -1, 1, 1, -1
    };

    BinarySVM model(
        100.0,
        SVMKernel::RBF,
        2.0,
        3,
        0.0,
        5000,
        1e-6,
        true,
        42
    );

    model.fit(X, y);

    REQUIRE(model.kernel() == SVMKernel::RBF);
    REQUIRE(model.gamma() == 2.0);
    REQUIRE(model.predict(X) == y);
}

TEST_CASE("BinarySVM with a polynomial kernel learns XOR",
          "[svm][binary][polynomial]") {
    Matrix X(4, 2);

    X(0, 0) = -1.0; X(0, 1) = -1.0;
    X(1, 0) = -1.0; X(1, 1) =  1.0;
    X(2, 0) =  1.0; X(2, 1) = -1.0;
    X(3, 0) =  1.0; X(3, 1) =  1.0;

    const std::vector<int> y = {
        -1, 1, 1, -1
    };

    BinarySVM model(
        100.0,
        SVMKernel::Polynomial,
        1.0,
        2,
        1.0,
        5000,
        1e-6,
        true,
        42
    );

    model.fit(X, y);

    REQUIRE(model.kernel() == SVMKernel::Polynomial);
    REQUIRE(model.degree() == 2);
    REQUIRE(model.coef0() == 1.0);
    REQUIRE(model.predict(X) == y);
}

TEST_CASE("BinarySVM computes automatic gamma from the feature count",
          "[svm][binary]") {
    Matrix X(4, 2);

    X(0, 0) = -2.0; X(0, 1) = -1.0;
    X(1, 0) = -1.0; X(1, 1) = -2.0;
    X(2, 0) =  1.0; X(2, 1) =  2.0;
    X(3, 0) =  2.0; X(3, 1) =  1.0;

    const std::vector<int> y = {
        -1, -1, 1, 1
    };

    BinarySVM model(
        1.0,
        SVMKernel::RBF,
        0.0
    );

    model.fit(X, y);

    // gamma = 1 / number_of_features = 1 / 2
    REQUIRE(
        model.gamma()
        == Catch::Approx(0.5).margin(1e-12)
    );
}

TEST_CASE("BinarySVM can train without an intercept",
          "[svm][binary]") {
    Matrix X(4, 1);

    X(0, 0) = -2.0;
    X(1, 0) = -1.0;
    X(2, 0) =  1.0;
    X(3, 0) =  2.0;

    const std::vector<int> y = {
        -1, -1, 1, 1
    };

    BinarySVM model(
        10.0,
        SVMKernel::Linear,
        0.0,
        3,
        0.0,
        5000,
        1e-6,
        false,
        42
    );

    model.fit(X, y);

    REQUIRE(
        model.intercept()
        == Catch::Approx(0.0).margin(1e-12)
    );

    REQUIRE(model.predict(X) == y);
}

TEST_CASE("BinarySVM rejects labels other than minus one and plus one",
          "[svm][binary]") {
    Matrix X(4, 1);

    X(0, 0) = -2.0;
    X(1, 0) = -1.0;
    X(2, 0) =  1.0;
    X(3, 0) =  2.0;

    // The public SVMClassifier will convert these labels to -1/+1.
    // The internal BinarySVM deliberately rejects them.
    const std::vector<int> y = {
        0, 0, 1, 1
    };

    BinarySVM model;

    REQUIRE_THROWS_AS(
        model.fit(X, y),
        std::invalid_argument
    );
}

TEST_CASE("BinarySVM requires both classes",
          "[svm][binary]") {
    Matrix X(3, 1);

    X(0, 0) = 1.0;
    X(1, 0) = 2.0;
    X(2, 0) = 3.0;

    const std::vector<int> y = {
        1, 1, 1
    };

    BinarySVM model;

    REQUIRE_THROWS_AS(
        model.fit(X, y),
        std::invalid_argument
    );
}

TEST_CASE("BinarySVM rejects incompatible training dimensions",
          "[svm][binary]") {
    Matrix X(3, 1);

    const std::vector<int> y = {
        -1, 1
    };

    BinarySVM model;

    REQUIRE_THROWS_AS(
        model.fit(X, y),
        std::invalid_argument
    );
}

TEST_CASE("BinarySVM requires fitting before prediction",
          "[svm][binary]") {
    BinarySVM model;
    Matrix X(2, 1);

    REQUIRE_THROWS_AS(
        model.predict(X),
        std::runtime_error
    );

    REQUIRE_THROWS_AS(
        model.decision_function(X),
        std::runtime_error
    );

    REQUIRE_THROWS_AS(
        model.support_vectors(),
        std::runtime_error
    );

    REQUIRE_THROWS_AS(
        model.dual_coefficients(),
        std::runtime_error
    );
}

TEST_CASE("BinarySVM rejects an incorrect number of prediction features",
          "[svm][binary]") {
    Matrix X_train(4, 1);

    X_train(0, 0) = -2.0;
    X_train(1, 0) = -1.0;
    X_train(2, 0) =  1.0;
    X_train(3, 0) =  2.0;

    const std::vector<int> y = {
        -1, -1, 1, 1
    };

    BinarySVM model(
        10.0,
        SVMKernel::Linear
    );

    model.fit(X_train, y);

    Matrix X_test(2, 2);

    REQUIRE_THROWS_AS(
        model.predict(X_test),
        std::invalid_argument
    );
}

TEST_CASE("BinarySVM rejects invalid constructor parameters",
          "[svm][binary]") {
    REQUIRE_THROWS_AS(
        BinarySVM(0.0),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        BinarySVM(
            1.0,
            SVMKernel::RBF,
            -1.0
        ),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        BinarySVM(
            1.0,
            SVMKernel::Polynomial,
            1.0,
            0
        ),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        BinarySVM(
            1.0,
            SVMKernel::RBF,
            1.0,
            3,
            0.0,
            0
        ),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        BinarySVM(
            1.0,
            SVMKernel::RBF,
            1.0,
            3,
            0.0,
            1000,
            0.0
        ),
        std::invalid_argument
    );
}