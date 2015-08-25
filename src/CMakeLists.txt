set(target "car_number_detector")

FILE(GLOB hdrs "*.h*" "${CAR_NUMBER_DETECT_INCLUDE}/*.h*")
FILE(GLOB srcs "*.cpp")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(${target} ${srcs} ${hdrs})
target_link_libraries(${target} ${LIBRARY_DEPS})

if(CMAKE_BUILD_TYPE MATCHES "Debug")
	message(STATUS "DEBUG!")
	set_property(TARGET ${target} APPEND PROPERTY COMPILE_DEFINITIONS "_DEBUG_")
endif()
