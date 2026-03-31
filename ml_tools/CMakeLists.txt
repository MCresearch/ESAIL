cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(nnof)

#list(APPEND CMAKE_PREFIX_PATH /path/to/Torch)
#list(APPEND libnpy_INCLUDE_DIR /path/to/libnpy)
macro(set_if_higher VARIABLE VALUE)
  if(${VARIABLE} LESS ${VALUE})
    set(${VARIABLE} ${VALUE})
  endif()
endmacro()

find_package(Torch REQUIRED)
set(CMAKE_CXX_STANDARD 14)
if(NOT Torch_VERSION VERSION_LESS "2.1.0")
    set_if_higher(CMAKE_CXX_STANDARD 17)
elseif(NOT Torch_VERSION VERSION_LESS "1.5.0")
    set_if_higher(CMAKE_CXX_STANDARD 14)
endif()

# Accept either a libnpy root path or an include path via -Dlibnpy_INCLUDE_DIR=...
find_path(libnpy_INCLUDE_PATH npy.hpp
    HINTS ${libnpy_INCLUDE_DIR}
    PATH_SUFFIXES include)
if(libnpy_INCLUDE_PATH)
    if(EXISTS "${libnpy_INCLUDE_PATH}/npy.hpp")
        set(libnpy_SOURCE_DIR "${libnpy_INCLUDE_PATH}/..")
    elseif(EXISTS "${libnpy_INCLUDE_PATH}/include/npy.hpp")
        set(libnpy_SOURCE_DIR "${libnpy_INCLUDE_PATH}")
    endif()
endif()

if(NOT libnpy_SOURCE_DIR)
    include(FetchContent)
    FetchContent_Declare(
        libnpy
        GIT_REPOSITORY https://github.com/llohse/libnpy.git
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE)
    FetchContent_MakeAvailable(libnpy)
else()
    message(STATUS "Using external libnpy from: ${libnpy_SOURCE_DIR}")
endif()

add_executable(nnof main.cpp data.cpp nn_of.cpp grid.cpp input.cpp kernel.cpp pauli_potential.cpp train_kedf.cpp)
target_include_directories(nnof PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${libnpy_SOURCE_DIR}/include)
target_link_libraries(nnof "${TORCH_LIBRARIES}")
set_target_properties(nnof PROPERTIES
    CXX_STANDARD ${CMAKE_CXX_STANDARD}
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)

#cmake -B build -Dlibnpy_INCLUDE_DIR=/path/to/libnpy/ -DTorch_DIR=/path/to/Torch/
#cmake -B build -Dlibnpy_INCLUDE_DIR=/path/to/libnpy/include/ -DTorch_DIR=/path/to/Torch/
