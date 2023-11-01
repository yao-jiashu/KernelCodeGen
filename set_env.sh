####################### Install MLIR  ###################################
git clone https://github.com/llvm/llvm-project.git
# select tag
git checkout llvmorg-15.0.0

cd llvm-project
mkdir build
cd build
cmake ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON

# n threads to compile
export N_THREADS=16
cmake --build .  -j ${N_THREADS} --target check-mlir
make -j ${N_THREADS} install
# ddd