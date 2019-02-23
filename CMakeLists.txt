cmake_minimum_required(VERSION 3.9)
project(LaneDetect)

# Looks for the spline header library. If not found, download it
find_path(SPLINEH ${CMAKE_CURRENT_SOURCE_DIR}/src/spline)
if(NOT $(SPLINEH))
    execute_process(COMMAND git clone https://github.com/ttk592/spline ${CMAKE_CURRENT_SOURCE_DIR}/src/spline)
endif()
# WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}

find_package(OpenCV REQUIRED)
set(OpenCV_INCLUDE_DIRS "/usr/include/opencv2")
message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

# Standard build type is Release. For debug mode, specify
# cmake [source-dir] -DCMAKE_BUILD_TYPE=Debug
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# set(SOURCES $(SOURCES) src/main.cpp src/LaneDetect.cpp)

add_executable(LaneDetect src/main.cpp src/LaneDetect.cpp)
target_link_libraries(LaneDetect ${OpenCV_LIBS})