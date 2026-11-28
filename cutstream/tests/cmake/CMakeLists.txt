cmake_minimum_required(VERSION 3.21)
project(cutstream_consumer LANGUAGES CXX)

if(DEFINED CUTSTREAM_SOURCE_DIR)
    add_subdirectory("${CUTSTREAM_SOURCE_DIR}" cutstream)
else()
    find_package(cutstream CONFIG REQUIRED)
endif()

add_executable(consumer ../test.cpp)
set_target_properties(consumer PROPERTIES CXX_STANDARD 11 CXX_EXTENSIONS OFF)
target_link_libraries(consumer PRIVATE cutstream::cutstream)
enable_testing()
add_test(NAME differential COMMAND consumer)
