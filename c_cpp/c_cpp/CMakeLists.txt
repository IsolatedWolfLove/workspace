cmake_minimum_required(VERSION 3.5.0)
project(c_cpp VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(OpenCV 4 REQUIRED)
add_executable(c_cpp src/f_test.cpp)

target_include_directories(c_cpp PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(c_cpp ${OpenCV_LIBS})