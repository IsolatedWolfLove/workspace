cmake_minimum_required(VERSION 3.5.0)
project(snake VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
find_package(OpenCV 4 REQUIRED)
add_executable(snake doubao_new_snack.cpp)

target_link_libraries(snake ${OpenCV_LIBS})