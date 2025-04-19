cmake_minimum_required (VERSION 3.10)
project(homoApp VERSION 0.0.1)
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_FLAGS "-g -O3") 

set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE include)
target_link_libraries(${PROJECT_NAME} INTERFACE)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)
add_subdirectory (test)