function(add_my_executable target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE)
    target_link_libraries(${target} ${PROJECT_NAME})
endfunction()

add_my_executable(test_doc test_doc.cpp)
add_my_executable(test_dec2quant test_dec2quant.cpp)
add_my_executable(test_dec2lorenzo test_dec2lorenzo.cpp)
