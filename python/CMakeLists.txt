include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# pybind11_add_module fait tout le travail specifique aux extensions
# Python : nommage du fichier de sortie (.pyd sur Windows, .so sur
# Linux/Mac), symboles exportes, lien avec l'interpreteur Python
# detecte automatiquement. Le nom donne ici ("chiikaml") est celui
# que Python utilisera pour "import chiikaml".
pybind11_add_module(chiikaml_python
    bindings.cpp
)

# Le module Python a besoin d'etre lie a notre bibliotheque C++ pour
# avoir acces a Matrix, KNNClassifier, KDTree, KMeans.
target_link_libraries(chiikaml_python PRIVATE chiikaml)

# "chiikaml_python" est le nom de la cible CMake, mais le fichier
# compile doit s'appeler exactement "chiikaml" (sans suffixe) pour que
# "import chiikaml" fonctionne en Python.
set_target_properties(chiikaml_python PROPERTIES OUTPUT_NAME chiikaml)
