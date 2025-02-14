function(add_my_executable target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE)
    target_link_libraries(${target} ${PROJECT_NAME})
endfunction()

# file(GLOB TEST_FILES ${TEST_DIR}/*.cpp)
# foreach(TEST_FILE ${TEST_FILES})
#     get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
#     add_my_executable(${TEST_NAME} ${TEST_FILE})
# endforeach(TEST_FILE)

add_my_executable(test_szx_derivative_2d test_szx_derivative_2d.cpp)
add_my_executable(test_szx_derivative_2d_nd test_szx_derivative_2d_nd.cpp)
# add_my_executable(test_szp_derivative_2d test_szp_derivative_2d.cpp)
# add_my_executable(test_szp_derivative_3d test_szp_derivative_3d.cpp)

# add_my_executable(test_szx_heatdis_2d test_szx_heatdis_2d.cpp)
