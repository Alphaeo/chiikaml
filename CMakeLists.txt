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

# --- Tests -------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

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
