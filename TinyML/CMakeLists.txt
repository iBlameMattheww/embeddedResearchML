cmake_minimum_required(VERSION 3.13)
project(TinyML C)

set(CMAKE_C_STANDARD 99)

include_directories(
    Target/include
    Target/tools
)

add_subdirectory(Target/src)
add_subdirectory(tests)