cmake_minimum_required(VERSION 3.20)
project(ten CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra)

add_library(ten STATIC
        src/builder/builder.cc
        src/ir/lower.cc
        src/codegen/lower.cc
        src/codegen/c_backend.cc
        src/runtime/runtime.cc
        src/scheduler/scheduler.h
        src/scheduler/tile_index.cc
)
target_include_directories(ten PUBLIC src)
target_link_libraries(ten dl)

add_executable(ten_tests tests/test.cc)
target_link_libraries(ten_tests ten)