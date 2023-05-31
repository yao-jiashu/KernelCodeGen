#include "IR/ComputeDAG/ComputeDAGDialect.h"
#include "IR/ComputeDAG/ComputeDAGOps.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::compute_dag;

#include "IR/ComputeDAG/ComputeDAGOpsDialect.cpp.inc"

void ComputeDAGDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "IR/ComputeDAG/ComputeDAGOps.cpp.inc"
      >();
}