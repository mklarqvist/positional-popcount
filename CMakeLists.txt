
cmake_minimum_required(VERSION 2.8)
project(libppospopcnt C CXX)
set(CMAKE_BUILD_TYPE Release)
include_directories(. benchmark/linux/)

set(CMAKE_CXX_STANDARD 11)  # enable C++11 standard
set(CMAKE_CXX_FLAGS "-O3 -march=native")
set(CMAKE_C_FLAGS "-O3 -march=native")

add_executable(bench pospopcnt.c benchmark.cpp)
add_executable(example pospopcnt.c example.c)
add_executable(instrumented_benchmark pospopcnt.c benchmark/linux/instrumented_benchmark.cpp)
add_executable(instrumented_benchmark_align64 pospopcnt.c benchmark/linux/instrumented_benchmark_align64.cpp)

enable_testing()
add_test(NAME example COMMAND example)
add_test(NAME bench COMMAND bench 100000 10)
add_test(NAME instrumented_benchmark COMMAND instrumented_benchmark -n 100000 -i 10)
add_test(NAME instrumented_benchmark_align64 COMMAND instrumented_benchmark_align64 -n 100000 -i 10)

install(FILES pospopcnt.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/pospopcnt)