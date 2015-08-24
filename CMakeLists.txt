CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(PROJECT_NAME car_number_detection)
project(${PROJECT_NAME})

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE})

set(CAR_NUMBER_DETECT_LIBRARY ${PROJECT_NAME})
set(CAR_NUMBER_DETECT_TESTS   "${PROJECT_NAME}_test")
set(CAR_NUMBER_DETECT_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")

include_directories("${CAR_NUMBER_DETECT_INCLUDE}")
# include_directories("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")

find_package(OpenCV REQUIRED)
set(LIBRARY_DEPS ${OpenCV_LIBS})

# BUILD
# add_subdirectory(3rdparty)
add_subdirectory(src)
# add_subdirectory(test)
# add_subdirectory(samples)

# REPORT
message( STATUS "")
message( STATUS "General configuration for ${PROJECT_NAME}")
message( STATUS "======================================")
message( STATUS "")
message( STATUS "   Configuration:        ${CMAKE_BUILD_TYPE}")
message( STATUS "   OpenCV build path:    ${OpenCV_DIR}")
message( STATUS "")
