cmake_minimum_required(VERSION 3.5.0)
project(my_homework_5 VERSION 0.1.0 LANGUAGES C CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 查找OpenCV库
find_package(OpenCV 4 REQUIRED)
find_package(CURL REQUIRED)
find_package(GDAL REQUIRED)
# 查找Python库
find_package(PythonLibs REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# 手动设置Python库和包含目录
set(PYTHON_INCLUDE_DIRS "/home/ccy/anaconda3/include/python3.12")
set(PYTHON_LIBRARIES "/home/ccy/anaconda3/lib/libpython3.12.so.1.0")

include_directories(${PYTHON_INCLUDE_DIRS})
link_directories(${Python_LIBRARY_DIRS})
# 添加NumPy头文件的路径
set(NUMPY_INCLUDE_DIRS "/home/ccy/anaconda3/lib/python3.12/site-packages/numpy/core/include")
include_directories(${NUMPY_INCLUDE_DIRS})
# 添加matplotlib-cpp库的包含目录
include_directories(/home/ccy/matplotlib-cpp/)

# 添加可执行文件
add_executable(my_homework_5 main.cpp)

# 链接OpenCV库
target_include_directories(my_homework_5 PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(my_homework_5 ${OpenCV_LIBS})

# 链接Python库
target_link_libraries(my_homework_5 ${PYTHON_LIBRARIES})

# 将Python包含目录添加到目标私有包含目录
target_include_directories(my_homework_5 PRIVATE ${Python_INCLUDE_DIRS})
target_include_directories(my_homework_5 PRIVATE ${NUMPY_INCLUDE_DIRS})
target_link_libraries(my_homework_5 ${CURL_LDFLAGS})
target_link_libraries(my_homework_5 ${GDAL_LDFLAGS})