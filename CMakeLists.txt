cmake_minimum_required(VERSION 3.10)

# Project name and version
project(Group_22 VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#Testing
include(CTest)
add_subdirectory(dependencies/googletest)

# Include directories
include_directories(src libs)

# Collect all source files in src
file(GLOB_RECURSE SRC_SOURCES "src/*.cpp")

# Add the executable
add_executable(main main.cpp ${SRC_SOURCES})

target_compile_options(main PRIVATE -Wall -Wextra)

# Define the path to the test data directory
set(PROJECT_TESTS_DIR "${CMAKE_SOURCE_DIR}/tests")

# Add a test executable for Google Test
file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")
if(TEST_SOURCES)
add_executable(tests ${TEST_SOURCES} ${SRC_SOURCES})

    # Link Google Test and the project libraries
    target_link_libraries(tests PRIVATE gtest gtest_main)

    # Define the test data path as a compile-time definition
    target_compile_definitions(tests PRIVATE PROJECT_TESTS_DIR="${PROJECT_TESTS_DIR}")

    # Enable CTest to discover and run Google Test cases
    include(GoogleTest)
    gtest_discover_tests(tests)

    # Add a debug build configuration
    add_executable(tests_debug ${TEST_SOURCES} ${SRC_SOURCES})
    target_compile_definitions(tests_debug PRIVATE DEBUG PROJECT_TESTS_DIR="${PROJECT_TESTS_DIR}")
    target_compile_definitions(tests_debug PRIVATE DEBUG)
    target_compile_options(tests_debug PRIVATE -g -O0 -Wall -Wextra)
    target_link_libraries(tests_debug PRIVATE gtest gtest_main)

    # Set debug executable properties
    set_target_properties(tests_debug PROPERTIES OUTPUT_NAME "tests_debug")
endif()
