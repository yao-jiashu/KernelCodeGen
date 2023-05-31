#include "IR/ComputeDAG/ComputeDAGOps.h"
#include "IR/ComputeDAG/ComputeDAGDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "IR/ComputeDAG/ComputeDAGOps.cpp.inc"




/////////////////////////////////////////////////
#include "mlir/IR/OpImplementation.h"



#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
//#include "mlir/Dialect/StandardOps/IR/Ops.h"
//#include "mlir/Dialect/StandardOps/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Operation.h"


mlir::LogicalResult mlir::compute_dag::GEMMOp::verify() {
  return mlir::success();
}

mlir::LogicalResult mlir::compute_dag::ReluOp::verify() {
  return mlir::success();
}