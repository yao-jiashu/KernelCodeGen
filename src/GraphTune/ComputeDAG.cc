#include "GraphTune/ComputeDAG.h"

namespace KernelCodegen {

ComputeDAG::DType ComputeDAG::getDataType(std::string dtype) {
  if (dtype == "float32") {
    return this->builder.getF32Type();
  }
  return nullptr;
}

ComputeDAG::Placholder ComputeDAG::placeholder(
  std::vector<int64_t> l, 
  std::string dtype,
  MemorySpace ms) {
  llvm::ArrayRef<int64_t> shape (l);
  auto dtype_ = getDataType(dtype);
  mlir::MemRefType tensorShape = mlir::MemRefType::get(
    shape, dtype_, {}, static_cast<int>(ms));
  return builder.create<ComputeDAG::Placholder>(builder.getUnknownLoc(), tensorShape);
}

ComputeDAG::GEMM ComputeDAG::gemm(ComputeDAG::Operand lhs, ComputeDAG::Operand rhs) {
  auto typeA = lhs->getResult(0).getType();
  auto typeB = rhs->getResult(0).getType();
  int64_t m {-1}, n {-1}, k1{-1}, k2{-1};
  ComputeDAG::DType dtype;
  mlir::Attribute memorySpace;
  if(typeA.isa<mlir::MemRefType>()) {
    auto shapeA = typeA.dyn_cast<mlir::MemRefType>();
    m = shapeA.getShape()[0];
    k1 = shapeA.getShape()[1];
    dtype = shapeA.getElementType();
    memorySpace = shapeA.getMemorySpace();
  }
  else {
    llvm::errs() << "Unsupported tensor type of A.\n";
    return nullptr;
  }
  if(typeB.isa<mlir::MemRefType>()) {
    auto shapeB = typeB.dyn_cast<mlir::MemRefType>();
    k2 = shapeB.getShape()[0];
    n = shapeB.getShape()[1];
  }
  else {
    llvm::errs() << "Unsupported tensor type of B.\n";
    return nullptr;
  }

  if (k1 != k2) {
    llvm::errs() << 
      "Can't apply GEMM Operation on with the input tensors due to imcompatible shape.\n";
    return nullptr;
  }
  std::vector<int64_t> l {m, n};
  llvm::ArrayRef<int64_t> shape (l);
  mlir::MemRefType CShape = mlir::MemRefType::get(
    shape, dtype, mlir::MemRefLayoutAttrInterface(), memorySpace);

  auto A = mlir::dyn_cast<mlir::Value>(lhs->getResult(0));
  auto B = mlir::dyn_cast<mlir::Value>(rhs->getResult(0));

  return builder.create<ComputeDAG::GEMM>(
    builder.getUnknownLoc(),
    CShape, A, B);
}

ComputeDAG::Relu ComputeDAG::relu(ComputeDAG::Operand input) {
  auto inputTensor = mlir::dyn_cast<mlir::Value>(input->getResult(0));
  auto outputShape = input->getResult(0).getType();
  return builder.create<ComputeDAG::Relu>(
    builder.getUnknownLoc(), 
    outputShape, inputTensor);
}

}