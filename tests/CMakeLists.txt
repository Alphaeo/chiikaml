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
)

target_link_libraries(chiikaml_tests PRIVATE
    chiikaml
    Catch2::Catch2WithMain
)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(chiikaml_tests)
