cmake_minimum_required(VERSION 3.23)
project(SyclFractalGenerator LANGUAGES CXX)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

find_package( OpenCV REQUIRED )
include_directories(src)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(SyclFractalGenerator
        src/main.cpp
        src/main.hpp)
target_link_libraries(SyclFractalGenerator sycl ${OpenCV_LIBS})
# target_compile_options(Mandelbrot PRIVATE -fsycl -fsycl-unnamed-lambda)
target_compile_options(SyclFractalGenerator PRIVATE -fsycl -fsycl-unnamed-lambda -fsycl-targets=nvptx64-nvidia-cuda)
# target_compile_options(Mandelbrot PRIVATE -fsycl -fsycl-unnamed-lambda -fsycl-targets=amd-gpu-gfx1034)
