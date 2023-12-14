#pragma once
#include "IR/IR.h"

namespace KernelCodeGen {

std::string CUDAGen(mlir::ModuleOp &module);

}