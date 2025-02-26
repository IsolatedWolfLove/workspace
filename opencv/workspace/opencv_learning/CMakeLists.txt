cmake_minimum_required(VERSION 3.5.0)
project(opencv_learning VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(OpenCV 4 REQUIRED)
add_executable(opencv_learning src/mind_2.cpp)
target_include_directories(opencv_learning PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv_learning ${OpenCV_LIBS})


