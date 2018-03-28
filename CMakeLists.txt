cmake_minimum_required(VERSION 3.10)
project(autodiff)
set(CMAKE_CXX_STANDARD 14)

add_executable(autodiff autodiff.cc)
find_package(benchmark REQUIRED)
target_include_directories(autodiff PUBLIC "${BENCHMARK_INCLUDE_DIRS}")
target_link_libraries(autodiff "benchmark")
set(CMAKE_CXX_FLAGS "-Ofast")
