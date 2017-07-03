cmake_minimum_required(VERSION 2.8.4)

project( opencv_test )

set(CMAKE_BUILD_TYPE Release)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(CUDA)
find_package(CUDA QUIET REQUIRED)


set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS};
    -O3 
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_62,code=sm_62
	#-gencode arch=compute_66,code=sm_66
)



#include_directories(/usr/local/include) 
#include_directories("/usr/local/cuda/include")
include_directories("/usr/local/cuda-8.0/include")
#include_directories(${CMAKE_BINARY_DIR})
#include_directories(/usr/include) 
#include_directories(/usr/include/x86_64-linux-gnu) 



#link_directories(/usr/lib)
#link_directories(/usr/lib/x86_64-linux-gnu)
#link_directories(/usr/local/cuda/lib64)
#link_directories(/usr/local/cuda-8.0/lib64)
#link_directories(/usr/local/cuda/targets/x86_64-linux/lib)

#LINK_DIRECTORIES(${CMAKE_BINARY_DIR})







add_executable( ${PROJECT_NAME} opencv_test.cpp)

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} nvcaffe_parser nvinfer)

#target_link_libraries( ${PROJECT_NAME} nvcaffe_parser nvinfer)













