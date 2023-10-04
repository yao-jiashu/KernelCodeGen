#include "IR/Schedule/ScheduleOps.h"
#include "IR/Schedule/ScheduleDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "IR/Schedule/ScheduleOps.cpp.inc"




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

using namespace mlir;
using namespace schedule;

namespace {

Value foldCastOp(Operation *op) {
    // Identity cast
    if (op->getOperand(0).getType() == op->getResult(0).getType())
        return op->getOperand(0);
    return nullptr;
}

LogicalResult
verifyCastOp(Operation *op,
                   function_ref<bool(Type, Type)> areCastCompatible) {
  auto opType = op->getOperand(0).getType();
  auto resType = op->getResult(0).getType();
  if (!areCastCompatible(opType, resType))
    return op->emitError("operand type ")
           << opType << " and result type " << resType
           << " are cast incompatible";

  return success();
}
}

//===----------------------------------------------------------------------===//
// VectorizeOp
//===----------------------------------------------------------------------===//

bool VectorizeOp::areCastCompatible(Type a, Type b) {
  return true;
  auto aT = a.dyn_cast<MemRefType>();
  auto bT = b.dyn_cast<MemRefType>();

  if (!aT || !bT)
    return false;

//  if (aT.getAffineMaps() != bT.getAffineMaps())
//    return false;

  if (aT.getMemorySpace() != bT.getMemorySpace())
    return false;

  // With rank 0, there is no vec cast.
  if (aT.getRank() == 0)
    return false;

  if (aT.getRank() != bT.getRank())
    return false;

  // Should have the same shape up until the last n-1 dimensions.
  if (!std::equal(aT.getShape().begin(), std::prev(aT.getShape().end()),
                  bT.getShape().begin()))
    return false;

  // The source memref can't have a vector elemental type.
  if (auto shapedEltType = aT.getElementType().dyn_cast<ShapedType>())
    return false;

  // The destination memref elt type has be a vector type.
  auto vectorEltTypeB = bT.getElementType().dyn_cast<VectorType>();
  if (!vectorEltTypeB)
    return false;

  auto eltA = aT.getElementType();
  auto eltB = vectorEltTypeB.getElementType();
  if (eltA != eltB)
    return false;

  int64_t lastDimA = aT.getShape().back();
  int64_t lastDimB = bT.getShape().back();

  // If one of them is dynamic but not the other, they are incompatible.
  if (lastDimA * lastDimB < 0)
    return false;

//  // The last dim of the target should be of the right size.
//  if (lastDimB != MemRefType::kDynamicSize &&
//      lastDimA / vectorEltTypeB.getNumElements() != lastDimB)
//    return false;

  return true;
}

mlir::LogicalResult VectorizeOp::verify() {
  return verifyCastOp(*this, areCastCompatible);
}

OpFoldResult VectorizeOp::fold(ArrayRef<Attribute> operands) {
  return foldCastOp(*this);
}

mlir::LogicalResult SliceVectorOp::verify() {
  return  mlir::success();
}

OpFoldResult SliceVectorOp::fold(ArrayRef<Attribute> operands) {
  return nullptr;
}