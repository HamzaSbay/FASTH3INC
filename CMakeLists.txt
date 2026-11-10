cmake_minimum_required(VERSION 3.20)
project(fasth3_in_c LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

find_package(OpenMP REQUIRED)

# Common compiler flags for GCC / Clang
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -std=gnu99
        -O3
        -Wall
        -Wextra
        -Wshadow
        -Wvla
        -march=native
        -fopenmp
        -ffp-contract=off
    )
endif()

# Core static library from src/** (core, io, model, tokenizer, vae, out, cli logic)
file(GLOB_RECURSE FH3_CORE_SRCS CONFIGURE_DEPENDS "src/*.c")
list(FILTER FH3_CORE_SRCS EXCLUDE REGEX "src/cli/fh3_run\\.c")

add_library(fh3_core STATIC ${FH3_CORE_SRCS})
target_include_directories(fh3_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(fh3_core PUBLIC OpenMP::OpenMP_C)

if(UNIX OR MINGW)
    target_link_libraries(fh3_core PUBLIC m)
endif()

if(WIN32)
    target_link_libraries(fh3_core PUBLIC psapi)
endif()

# CLI binary target
add_executable(fh3 src/cli/fh3_run.c)
target_link_libraries(fh3 PRIVATE fh3_core OpenMP::OpenMP_C)
if(UNIX OR MINGW)
    target_link_libraries(fh3 PRIVATE m)
endif()
if(WIN32)
    target_link_libraries(fh3 PRIVATE psapi)
endif()

# Benchmarks
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/bench_matmul.c")
    add_executable(bench_matmul benchmarks/bench_matmul.c)
    target_link_libraries(bench_matmul PRIVATE fh3_core OpenMP::OpenMP_C)
    if(UNIX OR MINGW)
        target_link_libraries(bench_matmul PRIVATE m)
    endif()
    if(WIN32)
        target_link_libraries(bench_matmul PRIVATE psapi)
    endif()
endif()

# CTest test suite
enable_testing()

file(GLOB_RECURSE TEST_SRCS CONFIGURE_DEPENDS "tests/unit/*.c")
foreach(test_file ${TEST_SRCS})
    get_filename_component(test_name_we ${test_file} NAME_WE)
    add_executable(${test_name_we} ${test_file})
    target_link_libraries(${test_name_we} PRIVATE fh3_core OpenMP::OpenMP_C)
    if(UNIX OR MINGW)
        target_link_libraries(${test_name_we} PRIVATE m)
    endif()
    if(WIN32)
        target_link_libraries(${test_name_we} PRIVATE psapi)
    endif()
    add_test(NAME ${test_name_we} COMMAND ${test_name_we} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()
