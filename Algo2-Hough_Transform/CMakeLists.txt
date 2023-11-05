cmake_minimum_required(VERSION 3.15)
project(houghTest)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find OpenMP package
find_package(OpenMP REQUIRED)

find_package(MPI)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

add_executable(houghTest main.cpp)

target_link_libraries(houghTest PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX ${MPI_C_LIBRARIES} ${MPI_LIBRARIES})
