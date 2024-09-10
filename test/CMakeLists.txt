function(add_my_executable target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE)
    target_link_libraries(${target} ${PROJECT_NAME})
endfunction()

add_my_executable(test_v1 test_v1.cpp)
add_my_executable(test_v2 test_v2.cpp)
add_my_executable(test_v3 test_v3.cpp)
