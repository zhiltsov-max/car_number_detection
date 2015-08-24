set(target ${CAR_NUMBER_DETECT_LIBRARY})

FILE(GLOB hdrs "*.h*" "${CAR_NUMBER_DETECT_INCLUDE}/*.h*")
FILE(GLOB srcs "*.cpp")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(${target} ${srcs} ${hdrs})
target_link_libraries(${target} ${LIBRARY_DEPS})
