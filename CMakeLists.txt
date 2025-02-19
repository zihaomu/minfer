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

message(STATUS "M_VERSION ${M_VERSION}")
message(STATUS "M_VERSION_MAJOR ${M_VERSION_MAJOR}")
project(minfer VERSION ${M_VERSION} LANGUAGES C CXX ASM)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    "${CMAKE_CURRENT_LIST_DIR}/cmake"
)

# Add debug macro to project
if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DM_DEBUG)
endif ()

# Load ROOT path and sent it to C++
add_definitions(-DM_ROOT_PATH="${CMAKE_CURRENT_LIST_DIR}")

include_directories(
        ${CMAKE_CURRENT_LIST_DIR}/include/
        ${CMAKE_CURRENT_LIST_DIR}/src/
        ${CMAKE_CURRENT_LIST_DIR}/src/core
        ${CMAKE_CURRENT_LIST_DIR}/tools/
        ${CMAKE_CURRENT_LIST_DIR}/3rdparty/
)

FILE(GLOB_RECURSE M_Core_SRC
        ${CMAKE_CURRENT_LIST_DIR}/src/core/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/core/gguf_mode/*.cpp
    )

add_library(M_Core OBJECT ${M_Core_SRC})
list(APPEND M_OBJECTS_TO_LINK $<TARGET_OBJECTS:M_Core>)
list(APPEND M_TARGETS M_Core)

include(src/backend/cpu/CMakeLists.txt)

include(test/CMakeLists.txt)