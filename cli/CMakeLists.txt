include(FetchContent)

FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.4.2
)
FetchContent_MakeAvailable(CLI11)

add_executable(chiikaml_cli main.cpp)

target_link_libraries(chiikaml_cli PRIVATE
    chiikaml
    CLI11::CLI11
)
