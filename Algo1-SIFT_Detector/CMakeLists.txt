cmake_minimum_required(VERSION 3.15)
project(opencvTest)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find OpenMP package
find_package(OpenMP REQUIRED)

message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

add_executable(opencvTest main.cpp)

target_link_libraries(opencvTest PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
