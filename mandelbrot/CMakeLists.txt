cmake_minimum_required(VERSION 3.15)

project(mandelbrot VERSION 0.1 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_LINK_EXECUTABLE "<CMAKE_C_COMPILER> <FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> /link <CMAKE_C_LINK_FLAGS> <LINK_FLAGS>")
set(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_CXX_COMPILER> <FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> /link <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS>")

find_package(OpenGL REQUIRED)

add_subdirectory(lib/glad)
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
add_subdirectory(lib/glfw)

add_executable(mandelbrot
    src/main.cpp src/gfx/display.cpp src/gfx/display.hpp src/gfx/renderer.cpp src/gfx/renderer.hpp
    src/cuda_selector.hpp)
target_link_libraries(mandelbrot glad OpenGL::GL glfw)
