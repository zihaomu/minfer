cmake_minimum_required(VERSION 3.10)

# Versioning stuff
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/minfer/define.h" M_DEFINE)
string(REGEX MATCH "M_VERSION_MAJOR [0-9]+" M_VERSION_MAJOR_DEFINE ${M_DEFINE})
string(REGEX MATCH "[0-9]+" M_VERSION_MAJOR ${M_VERSION_MAJOR_DEFINE})
string(REGEX MATCH "M_VERSION_MINOR [0-9]+" M_VERSION_MINOR_DEFINE ${M_DEFINE})
string(REGEX MATCH "[0-9]+" M_VERSION_MINOR ${M_VERSION_MINOR_DEFINE})
string(REGEX MATCH "M_VERSION_PATCH [0-9]+" M_VERSION_PATCH_DEFINE ${M_DEFINE})
string(REGEX MATCH "[0-9]+" M_VERSION_PATCH ${M_VERSION_PATCH_DEFINE})
set(M_VERSION ${M_VERSION_MAJOR}.${M_VERSION_MINOR}.${M_VERSION_PATCH})

# default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "M_VERSION ${M_VERSION}")
message(STATUS "M_VERSION_MAJOR ${M_VERSION_MAJOR}")
project(minfer VERSION ${M_VERSION} LANGUAGES C CXX ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    "${CMAKE_CURRENT_LIST_DIR}/cmake"
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: enabling parallel kernels")
    add_compile_definitions(M_HAS_OPENMP=1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(STATUS "OpenMP not found: kernels remain single-threaded")
endif()

# Add debug macro to project
if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DM_DEBUG)
endif ()

# Load ROOT path and sent it to C++
add_definitions(-DM_ROOT_PATH="${CMAKE_CURRENT_LIST_DIR}")

include_directories(
        ${CMAKE_CURRENT_LIST_DIR}/3rdparty/
        ${CMAKE_CURRENT_LIST_DIR}/3rdparty/mobilekv/include
        ${CMAKE_CURRENT_LIST_DIR}/3rdparty/xsimd/include
        ${CMAKE_CURRENT_LIST_DIR}/include/
        ${CMAKE_CURRENT_LIST_DIR}/src/
        ${CMAKE_CURRENT_LIST_DIR}/src/core
        ${CMAKE_CURRENT_LIST_DIR}/tools/
)

FILE(GLOB_RECURSE M_Core_SRC
        ${CMAKE_CURRENT_LIST_DIR}/src/core/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/core/gguf_mode/*.cpp
    )
list(APPEND M_Core_SRC
        ${CMAKE_CURRENT_LIST_DIR}/3rdparty/mobilekv/src/kv_cache.cpp
    )

add_library(M_Core OBJECT ${M_Core_SRC})
list(APPEND M_OBJECTS_TO_LINK $<TARGET_OBJECTS:M_Core>)
list(APPEND M_TARGETS M_Core)

include(src/backend/cpu/CMakeLists.txt)

include(test/CMakeLists.txt)

# Chat example
add_executable(chat ${CMAKE_CURRENT_LIST_DIR}/example/chat.cpp)
target_link_libraries(chat ${M_TARGETS})

# LLM benchmark
add_executable(minfer_benchmark ${CMAKE_CURRENT_LIST_DIR}/benchmark/llm_benchmark.cpp)
target_link_libraries(minfer_benchmark ${M_TARGETS})

# Operator benchmark
add_executable(minfer_op_benchmark ${CMAKE_CURRENT_LIST_DIR}/benchmark/op_benchmark.cpp)
target_link_libraries(minfer_op_benchmark ${M_TARGETS})
