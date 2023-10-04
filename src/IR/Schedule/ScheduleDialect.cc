#include "IR/Schedule/ScheduleDialect.h"
#include "IR/Schedule/ScheduleOps.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::schedule;

#include "IR/Schedule/ScheduleOpsDialect.cpp.inc"

void ScheduleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "IR/Schedule/ScheduleOps.cpp.inc"
      >();
}