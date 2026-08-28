include(FetchContent)

# Google Benchmark construit par defaut ses propres tests (qui tirent
# en plus GoogleTest comme dependance) -- on les desactive, on ne
# veut que la bibliotheque de benchmark elle-meme.
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.1
)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(chiikaml_benchmarks
    bench_knn_vs_kdtree.cpp
    bench_random_forest_threading.cpp
)

target_link_libraries(chiikaml_benchmarks PRIVATE
    chiikaml
    benchmark::benchmark
)
