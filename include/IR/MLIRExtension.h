#pragma once
#include "IR/MLIR.h"
#include "IR/ComputeDAG/ComputeDAGDialect.h"
#include "IR/ComputeDAG/ComputeDAGOps.h"

namespace KernelCodegen {
  
using Context = mlir::MLIRContext;

inline void initContext(Context & context) {
  context.getOrLoadDialect<mlir::compute_dag::ComputeDAGDialect>();
  context.getOrLoadDialect<mlir::AffineDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::arith::ArithmeticDialect>();
  context.getOrLoadDialect<mlir::gpu::GPUDialect>();
  context.getOrLoadDialect<mlir::vector::VectorDialect>();
  mlir::registerAllPasses();
}

}