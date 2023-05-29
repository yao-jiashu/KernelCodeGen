#pragma once
#include "MLIR.h"

// KernelCodegen Dialect
#include "GraphTune/ComputeDAG.h"

namespace KernelCodegen {

// API for init MLIR env
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