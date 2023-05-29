#include "GraphTune/ComputeDAG/ComputeDAGDialect.h"
#include "GraphTune/ComputeDAG/ComputeDAGOps.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::compute_dag;

#include "GraphTune/ComputeDAG/ComputeDAGOpsDialect.cpp.inc"

void ComputeDAGDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "GraphTune/ComputeDAG/ComputeDAGOps.cpp.inc"
      >();
}