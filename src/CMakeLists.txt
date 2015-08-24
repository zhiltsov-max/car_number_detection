set(target ${CAR_NUMBER_DETECT_LIBRARY})

FILE(GLOB hdrs "*.h*" "${CAR_NUMBER_DETECT_INCLUDE}/*.h*")
FILE(GLOB srcs "*.cpp")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

ADD_LIBRARY(${target} STATIC ${srcs} ${hdrs})
TARGET_LINK_LIBRARIES(${target} ${LIBRARY_DEPS})
