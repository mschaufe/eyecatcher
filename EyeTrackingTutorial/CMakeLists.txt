find_package(OpenCV REQUIRED) # after project()

cmake_minimum_required(VERSION 3.15)
project(EyeTrackingTutorial)

set(CMAKE_CXX_STANDARD 17)

add_executable(EyeTrackingTutorial main.cpp)

#At the end
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(EyeTrackingTutorial ${OpenCV_LIBS})