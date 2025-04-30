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

add_my_executable(divergence_szx_3d divergence_szx_3d.cpp)
add_my_executable(laplacian_szx_3d laplacian_szx_3d.cpp)
add_my_executable(curl_szx_3d curl_szx_3d.cpp)
add_my_executable(derivative_szx_3d derivative_szx_3d.cpp)


# add_my_executable(laplacian_szp_3d laplacian_szp_3d.cpp)
# add_my_executable(decompress_szp_3d decompress_szp_3d.cpp)
# add_my_executable(decompress_szx_3d decompress_szx_3d.cpp)
# add_my_executable(curl_szp_3d curl_szp_3d.cpp)
# add_my_executable(divergence_szp_3d divergence_szp_3d.cpp)

# add_my_executable(laplacian_szx_3d laplacian_szx_3d.cpp)

# add_my_executable(decompress_szp_2d decompress_szp_2d.cpp)
# add_my_executable(derivative_szp_3d derivative_szp_3d.cpp)

# add_my_executable(variance_szx_3d variance_szx_3d.cpp)
# add_my_executable(variance_szx_2d variance_szx_2d.cpp)
# add_my_executable(variance_szx_1d variance_szx_1d.cpp)

# add_my_executable(variance_szp_3d variance_szp_3d.cpp)
# add_my_executable(variance_szp_2d variance_szp_2d.cpp)
# add_my_executable(variance_szp_1d variance_szp_1d.cpp)
