#Unit test
set(gtest_SOURCE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/googletest")
add_subdirectory(${gtest_SOURCE_DIR})
enable_testing()

include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${gtest_SOURCE_DIR}
        ${gtest_SOURCE_DIR}/include
#        ${CMAKE_SOURCE_DIR}/test/opencv_lab/mat_test
)

file(GLOB OPENCV_MAT_SRC
        ${CMAKE_SOURCE_DIR}/test/opencv_lab/mat_test/*.cpp
)

#add_library(opencv_lab ${OPENCV_MAT_SRC})

file(GLOB TEST_SRC
        ${CMAKE_SOURCE_DIR}/test/*.cpp
        ${CMAKE_SOURCE_DIR}/test/core/*.cpp              # 包含核心功能测试
        ${CMAKE_SOURCE_DIR}/test/layers/*.cpp            # 包含层实现测试和层数据生成
        ${CMAKE_SOURCE_DIR}/test/nets/*.cpp            # 包含常见llm模型测试
        #        ${CMAKE_SOURCE_DIR}/test/opencv_lab/mat_test.cpp # 包含mat实验相关测试
)

add_executable(minfer_test ${TEST_SRC})
target_link_libraries(minfer_test gtest gtest_main)
target_link_libraries(minfer_test ${M_TARGETS})

# Link sentencepiece_DIR
set(sentencepiece_DIR "/Users/moo/workssd/github/sentencepiece/build/src")
#find_package(sentencepiece REQUIRED)
include_directories("/Users/moo/workssd/github/sentencepiece/src")
target_link_libraries(minfer_test "${sentencepiece_DIR}/libsentencepiece.dylib")

# This is so you can do 'make test' to see all your tests run, instead of
# manually running the executable runUnitTests to see those specific tests.
add_test(NAME minfer_test COMMAND `minfer_test`)