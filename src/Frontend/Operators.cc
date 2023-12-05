#include "Frontend/Operators.h"

namespace KernelCodeGen {

mlir::Type getDType(mlir::OpBuilder& builder, const std::string& dtype) {
  if(dtype == "float32") return builder.getF32Type();
  if(dtype == "float64") return builder.getF64Type();
  if(dtype == "float16") return builder.getF16Type();
  return nullptr;
}

std::string toStr(mlir::Type type) {
  if(type.isa<mlir::Float16Type>()) return {"float16"};
  if(type.isa<mlir::Float32Type>()) return {"float32"};
  if(type.isa<mlir::Float64Type>()) return {"float64"};
  return nullptr;
}

mlir::Value PlaceHolder::build(ComputeDAG* graph, 
    const std::vector<int64_t>& shapes, 
    const std::string& dtype) {
  auto builder = graph->builder;
//   llvm::ArrayRef<int64_t> shapes_ (shapes);
  auto dtype_ = getDType(builder, dtype);
  auto tType = mlir::MemRefType::get(
    shapes, dtype_, {}, static_cast<int>(MemorySpace::global));
  auto allocOp = builder.create<mlir::memref::AllocOp>(
    builder.getUnknownLoc(), tType);
  return allocOp.getResult();
}

mlir::Value Matmul::build(ComputeDAG* graph, mlir::Value A, mlir::Value B, const std::string& dtype_) {
  auto builder = graph->builder;
  auto typeA = A.getType();
  auto typeB = B.getType();
  int64_t m {-1}, n {-1}, k1{-1}, k2{-1};
  mlir::Attribute memorySpace;
  mlir::Type elementTypeA;

  if(typeA.isa<mlir::MemRefType>()) {
    auto shapeA = typeA.dyn_cast<mlir::MemRefType>();
    m = shapeA.getShape()[0];
    k1 = shapeA.getShape()[1];
    elementTypeA = shapeA.getElementType();
    memorySpace = shapeA.getMemorySpace();
  }
  else {
    llvm::errs() << "Type of left operand of Matmul is not Memref.\n";
    return nullptr;
  }
  auto dtype = dtype_ != ""  ? dtype_ : toStr(elementTypeA);

  if(typeB.isa<mlir::MemRefType>()) {
    auto shapeB = typeB.dyn_cast<mlir::MemRefType>();
    k2 = shapeB.getShape()[0];
    n = shapeB.getShape()[1];
  }
  else {
    llvm::errs() << "Type of right operand of Matmul is not Memref.\n";
    return nullptr;
  }

  if (k1 != k2) {
    llvm::errs() << 
      "Can't apply Matmul Operation due to imcompatible K-dim.\n";
    return nullptr;
  }

  // Create C buffer as the result.
  auto C = graph->create<PlaceHolder>(std::vector<int64_t>{m, n}, dtype);

  // void buildAffineLoopNest(OpBuilder &builder, Location loc,
  //                         ArrayRef<int64_t> lbs, ArrayRef<int64_t> ubs,
  //                         ArrayRef<int64_t> steps,
  //                         function_ref<void(OpBuilder &, Location, ValueRange)>
  //                             bodyBuilderFn = nullptr);
  mlir::SmallVector<int64_t, 3> lowerBounds(2, /*Value=*/0);
  mlir::SmallVector<int64_t, 3> steps(2, /*Value=*/1);
  mlir::SmallVector<int64_t, 3> upperBounds({m, n});
  mlir::buildAffineLoopNest(
    builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
      auto i = ivs[0];
      auto j = ivs[1];
      // FloatAttr Builder::getFloatAttr(Type type, double value) {
      //   return FloatAttr::get(type, value);
      // }
      // initilize to 0
      auto dtypeC = getDType(nestedBuilder, dtype);
      auto zero = nestedBuilder.create<mlir::arith::ConstantOp>(nestedBuilder.getUnknownLoc(), 
          nestedBuilder.getFloatAttr(dtypeC, 0));

      auto kLoopBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv,
                          mlir::ValueRange iterArgs) {
        mlir::OpBuilder::InsertionGuard nestedGuard(builder);
        auto k = iv;
        auto ld_a = builder.create<mlir::AffineLoadOp>(
                      builder.getUnknownLoc(), A, mlir::ValueRange({i, k}));
        auto ld_b = builder.create<mlir::AffineLoadOp>(
                      builder.getUnknownLoc(), B, mlir::ValueRange({k, j}));
        auto mul = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), ld_a, ld_b);
        auto add = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), mul, iterArgs[0]);
        builder.create<mlir::AffineYieldOp>(builder.getUnknownLoc(), add.getResult());
      };
      auto Cij = nestedBuilder.create<mlir::AffineForOp>(nestedBuilder.getUnknownLoc(), 
        0, k1, 1, /*iterArgs=lvm::None*/ mlir::ValueRange({zero.getResult()}), kLoopBody);

      nestedBuilder.create<mlir::AffineStoreOp>(nestedBuilder.getUnknownLoc(), 
          Cij.getResult(0), C, mlir::ValueRange({i, j}));
    }
  );
  return C;
}


mlir::Value Relu::build(ComputeDAG* graph, mlir::Value input, const std::string& dtype_) {
  auto builder = graph->builder;
  auto type = input.getType();

  mlir::Attribute memorySpace;
  mlir::Type elementType;

  llvm::ArrayRef<int64_t> shape;

  if(type.isa<mlir::MemRefType>()) {
    auto type_ = type.dyn_cast<mlir::MemRefType>();
    shape = type_.getShape();
    elementType = type_.getElementType();
    memorySpace = type_.getMemorySpace();
  }
  else {
    llvm::errs() << "Type of input of Relu is not Memref.\n";
    return nullptr;
  }
  auto dtype = dtype_ != ""  ? dtype_ : toStr(elementType);

  mlir::SmallVector<int64_t, 8> lowerBounds(shape.size(), /*Value=*/0);
  mlir::SmallVector<int64_t, 8> steps(shape.size(), /*Value=*/1);
  mlir::SmallVector<int64_t, 8> upperBounds(shape.begin(), shape.end());
  mlir::buildAffineLoopNest(
    builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {

      // initilize to 0
      auto dtypeOutput = getDType(nestedBuilder, dtype);
      auto zero = nestedBuilder.create<mlir::arith::ConstantOp>(nestedBuilder.getUnknownLoc(), 
          nestedBuilder.getFloatAttr(dtypeOutput, 0));
      auto ld_element = nestedBuilder.create<mlir::AffineLoadOp>(nestedBuilder.getUnknownLoc(), input, ivs);
      auto max = nestedBuilder.create<mlir::arith::MaxFOp>(nestedBuilder.getUnknownLoc(), zero, ld_element);
      nestedBuilder.create<mlir::AffineStoreOp>(nestedBuilder.getUnknownLoc(), max, input, ivs);
    }
  );
  return input;
}

}