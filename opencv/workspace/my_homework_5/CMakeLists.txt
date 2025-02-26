cmake_minimum_required(VERSION 3.5.0)
project(my_homework_5 VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(OpenCV 4 REQUIRED)
add_executable(my_homework_5 main.cpp)

target_include_directories(my_homework_5 PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(my_homework_5 ${OpenCV_LIBS})
