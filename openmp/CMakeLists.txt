cmake_minimum_required(VERSION 2.8)
project( erosion_openmp )
find_package( OpenCV REQUIRED )
add_executable( erosion_openmp erosion_openmp.cc )
target_link_libraries( erosion_openmp ${OpenCV_LIBS} )