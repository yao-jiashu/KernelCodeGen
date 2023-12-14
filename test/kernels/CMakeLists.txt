include_directories(${CUDA_MATH_LIBS_PATH}/include)
include_directories(.)

add_executable(matmul matmul.cu)
set_target_properties(matmul PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH_VAR})
target_compile_options(matmul PRIVATE "-O2" --gpu-architecture=compute_80 -lineinfo)
target_link_libraries(matmul PRIVATE
        ${CUDA_MATH_LIBS_PATH}/lib64/libcublas.so)
