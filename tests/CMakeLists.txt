include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.2
)
FetchContent_MakeAvailable(Catch2)

add_executable(chiikaml_tests
    test_matrix.cpp
    test_knn.cpp
    test_kdtree.cpp
    test_kmeans.cpp
    test_decision_tree.cpp
    test_random_forest.cpp
    test_linear_regression.cpp
    test_ridge_regression.cpp
    test_lasso_regression.cpp
    test_tensor.cpp
    test_classification_metrics.cpp
    test_split.cpp
)

target_link_libraries(chiikaml_tests PRIVATE
    chiikaml
    Catch2::Catch2WithMain
)

# Chemin absolu (calcule a la compilation) vers tests/data/, pour que
# les tests trouvent les fichiers CSV peu importe le repertoire de
# travail depuis lequel ctest/l'executable est lance.
target_compile_definitions(chiikaml_tests PRIVATE
    CHIIKAML_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(chiikaml_tests)
