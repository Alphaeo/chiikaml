cmake_minimum_required(VERSION 3.20)
project(cppml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- La bibliotheque cppml -------------------------------------------------
add_library(cppml
    src/matrix.cpp
)

target_include_directories(cppml PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(cppml PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(cppml PUBLIC /W4)
else()
    target_compile_options(cppml PUBLIC -Wall -Wextra -Wpedantic)
endif()

# --- Tests -------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
