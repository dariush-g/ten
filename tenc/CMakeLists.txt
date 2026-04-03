cmake_minimum_required(VERSION 3.20)
project(ten CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra)

set(LIB_SRC
        src/builder/builder.cc
        src/ir/lower.cc
        src/codegen/lower.cc
        src/codegen/c_backend.cc
        src/runtime/runtime.cc
)

add_executable(tenc src/main.cc ${LIB_SRC})
target_include_directories(tenc PRIVATE src)
target_link_libraries(tenc dl)