include(FetchContent)

FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.18.5
)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(httplib nlohmann_json)

# chiikaml_viz est une bibliotheque a part, separee de chiikaml :
# le coeur C++ (Matrix, KNNClassifier, KDTree, KMeans...) ne doit
# jamais dependre d'un serveur HTTP/JSON, meme optionnellement --
# meme principe que pour les bindings Python (CHIIKAML_BUILD_PYTHON).
add_library(chiikaml_viz
    src/dashboard.cpp
    src/visualize_kmeans.cpp
)

target_include_directories(chiikaml_viz PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(chiikaml_viz PUBLIC
    chiikaml
    httplib::httplib
    nlohmann_json::nlohmann_json
)

target_compile_features(chiikaml_viz PUBLIC cxx_std_20)

add_executable(chiikaml_kmeans_viz_demo examples/kmeans_demo.cpp)
target_link_libraries(chiikaml_kmeans_viz_demo PRIVATE chiikaml_viz)
