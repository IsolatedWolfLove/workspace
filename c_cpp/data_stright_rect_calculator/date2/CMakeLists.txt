cmake_minimum_required(VERSION 3.5.0)
project(my_homework_5 VERSION 0.1.0 LANGUAGES C CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 查找OpenCV库
find_package(OpenCV 4 REQUIRED)

# 添加matplotlib-cpp库的包含目录
include_directories(/home/ccy/matplotlib-cpp/)

# 添加可执行文件
add_executable(my_homework_5 main.cpp)

# 链接OpenCV库
target_include_directories(my_homework_5 PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(my_homework_5 ${OpenCV_LIBS})
