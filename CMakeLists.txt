cmake_minimum_required(VERSION 3.20)
project(chiikaml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- La bibliotheque chiikaml -------------------------------------------------
add_library(chiikaml
    src/matrix.cpp
    src/knn.cpp
    src/kdtree.cpp
    src/kmeans.cpp
    src/decision_tree.cpp
    src/random_forest.cpp
    src/linear_regression.cpp
    src/ridge_regression.cpp
    src/lasso_regression.cpp
    src/tensor.cpp
    src/classification_metrics.cpp
    src/train_test_split.cpp
    src/binary_svm.cpp
)

target_include_directories(chiikaml PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(chiikaml PUBLIC cxx_std_20)

# std::thread a besoin d'etre explicitement lie a la bibliotheque
# threads de la plateforme (pthread sous MinGW/Linux) -- sans ca,
# erreur de lien a la compilation de random_forest.cpp.
find_package(Threads REQUIRED)
target_link_libraries(chiikaml PUBLIC Threads::Threads)

if(MSVC)
    target_compile_options(chiikaml PUBLIC /W4)
else()
    target_compile_options(chiikaml PUBLIC -Wall -Wextra -Wpedantic)
endif()

# Desactive par defaut : -mavx2 genere du code qui plante ("illegal
# instruction") sur un CPU qui ne supporte pas AVX2 (grosso modo,
# avant ~2013). A activer explicitement (-DCHIIKAML_ENABLE_AVX2=ON)
# si on sait que la machine cible le supporte -- ce qui est le cas de
# la grande majorite des machines aujourd'hui, mais ce n'est pas une
# garantie universelle comme -O3 l'est.
option(CHIIKAML_ENABLE_AVX2 "Compile with AVX2 SIMD instructions (requires a CPU from ~2013+)" OFF)
if(CHIIKAML_ENABLE_AVX2)
    if(MSVC)
        target_compile_options(chiikaml PUBLIC /arch:AVX2)
    else()
        target_compile_options(chiikaml PUBLIC -mavx2)
    endif()
endif()

# --- Tests -------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

# --- Benchmarks (Phase 7) ------------------------------------------------
# Contrairement a CHIIKAML_BUILD_PYTHON/CHIIKAML_BUILD_VIZ, actif par
# defaut : Google Benchmark est une dependance C++ normale (comme
# Catch2), pas une extension vers un autre ecosysteme -- et mesurer la
# perf fait partie du travail de la Phase 7 en cours, pas une option
# annexe.
option(CHIIKAML_BUILD_BENCHMARKS "Build Google Benchmark microbenchmarks" ON)
if(CHIIKAML_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# --- CLI (Phase 8) --------------------------------------------------------
# Actif par defaut, meme raisonnement que CHIIKAML_BUILD_BENCHMARKS :
# CLI11 est une dependance C++ normale (header-only), pas une
# extension vers un autre ecosysteme.
option(CHIIKAML_BUILD_CLI "Build the chiikaml_cli command-line tool" ON)
if(CHIIKAML_BUILD_CLI)
    add_subdirectory(cli)
endif()

# --- Bindings Python (optionnel) ----------------------------------------
# Desactive par defaut : le coeur C++ (bibliotheque + tests) ne doit
# jamais dependre de Python. On ne construit ce sous-dossier que si on
# le demande explicitement.
option(CHIIKAML_BUILD_PYTHON "Build the pybind11 Python bindings" OFF)
if(CHIIKAML_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# --- Dashboard de visualisation (optionnel) ------------------------------
# Meme principe que CHIIKAML_BUILD_PYTHON : desactive par defaut, le
# coeur chiikaml ne doit jamais dependre d'un serveur HTTP/JSON.
option(CHIIKAML_BUILD_VIZ "Build the local visualization dashboard" OFF)
if(CHIIKAML_BUILD_VIZ)
    add_subdirectory(viz)
endif()
