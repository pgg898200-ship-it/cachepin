cmake_minimum_required(VERSION 3.21)
project(cutstream LANGUAGES CXX)

add_library(cutstream INTERFACE)
add_library(cutstream::cutstream ALIAS cutstream)
target_compile_features(cutstream INTERFACE cxx_std_17)
target_include_directories(cutstream INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")

if(PROJECT_IS_TOP_LEVEL)
    include(GNUInstallDirs)
    target_include_directories(cutstream INTERFACE
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
    install(FILES include/cutstream.hpp DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
    install(TARGETS cutstream EXPORT cutstreamTargets)
    install(EXPORT cutstreamTargets FILE cutstreamConfig.cmake
        NAMESPACE cutstream:: DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cutstream")
endif()
