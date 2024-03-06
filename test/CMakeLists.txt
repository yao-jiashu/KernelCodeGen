add_executable(codegen_graph test.cc)
target_link_libraries(codegen_graph PUBLIC kcg_runtime)

add_subdirectory(matmul)