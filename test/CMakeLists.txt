#Unit test
set(gtest_SOURCE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/googletest")
add_subdirectory(${gtest_SOURCE_DIR})
enable_testing()

include_directories(${gtest_SOURCE_DIR} ${gtest_SOURCE_DIR}/include)

file(GLOB TEST_SRC
        ${CMAKE_SOURCE_DIR}/test/*.cpp
        ${CMAKE_SOURCE_DIR}/test/core/*.cpp
)

add_executable(minfer_test ${TEST_SRC})
target_link_libraries(minfer_test gtest gtest_main)
target_link_libraries(minfer_test ${M_TARGETS})

# This is so you can do 'make test' to see all your tests run, instead of
# manually running the executable runUnitTests to see those specific tests.
add_test(NAME minfer_test COMMAND minfer_test)