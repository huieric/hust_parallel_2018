cmake_minimum_required(VERSION 2.8)
project( erosion_mpi )
find_package( OpenCV REQUIRED )
find_package( MPI REQUIRED )
include_directories(${MPI_INCLUDE_PATH})
add_executable( erosion_mpi erosion_mpi.cc )
target_link_libraries( erosion_mpi ${OpenCV_LIBS} ${MPI_LIBRARIES} )

if(MPI_COMPILE_FLAGS)
  set_target_properties(erosion_mpi PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(erosion_mpi PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()