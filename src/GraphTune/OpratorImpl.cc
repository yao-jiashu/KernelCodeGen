#include "MLIREnhance.h"
#include <memory>
#include <vector>

// build a rule to implement one op
// for example, the outer loop num is related to output's shape
// the inner loop's num is related to reduction operation
// based on this rule, we can implement OperatorFuse
namespace {
using namespace mlir;
using namespace KernelCodegen;
struct GEMMImplement : 
  public PassWrapper<GEMMImplement, OperationPass<ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GEMMImplement)
   GEMMImplement() {}
   void runOnOperation() override;
};

void GEMMImplement::runOnOperation() {
   ModuleOp module = getOperation();
   if (module->hasAttr("compute_dag.gemm_kernel")) {
    return;
   }
   // The walker proceeds in pre-order to process
   module.walk<WalkOrder::PreOrder>([&](compute_dag::GEMMOp gemmOp) {
    auto outType = gemmOp.getResult().getType();
    int64_t m {-1}, n {-1};
    Type dtype;
    if(outType.isa<MemRefType>()) {
      auto outShape = outType.dyn_cast<MemRefType>();
      m = outShape.getShape()[0];
      n = outShape.getShape()[1];
      dtype = outShape.getElementType();
    }
    else {
      llvm::errs() << "Unsupported tensor type of the output of the GEMM.";
      return;
    }

    auto typeA = gemmOp.getOperands()[0].getType();
    auto typeB = gemmOp.getOperands()[1].getType();
    int64_t k {-1};
    if(typeA.isa<MemRefType>()) {
      auto shapeA = typeA.dyn_cast<MemRefType>();
      k = shapeA.getShape()[1];
    }
    else {
      llvm::errs() << "Unsupported tensor type of the left operand.";
      return;
    }

    OpBuilder builder(module.getContext());
    module->setAttr(std::string("compute_dag.gemm_kernel"), 
      builder.getStringAttr("True"));

    builder.setInsertionPointToEnd(module.getBody());
    OpBuilder::InsertionGuard guard(builder);

    auto int32Type = builder.getI32Type();
    std::vector<Type> typesArray 
      {outType, typeA, typeB, int32Type, int32Type, int32Type};
    ArrayRef<Type> paramTypes(typesArray);
    auto functionType = builder.getFunctionType(TypeRange(paramTypes), llvm::None);

    auto func = builder.create<func::FuncOp>(
      builder.getUnknownLoc(), StringRef("gemm_kernel"), functionType);

    func->getRegion(0).push_back(new Block);
    Block &bodyBlock = func.front();
    int nums = static_cast<int>(paramTypes.size());
    for (int i = 0; i < nums; i++ ) {
      bodyBlock.addArguments(paramTypes[i], builder.getUnknownLoc());
    }
    ValueRange operands = bodyBlock.getArguments();
    builder.setInsertionPointToStart(&bodyBlock);  
  
    // // build loops
    SmallVector<int64_t, 3> lowerBounds(3, /*Value=*/0);
    SmallVector<int64_t, 3> steps(3, /*Value=*/1);
    SmallVector<int64_t, 3> upperBounds({m, n, k});
    buildAffineLoopNest(
      builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        auto C = operands[0];
        auto A = operands[1];
        auto B = operands[2];
        auto i = ivs[0];
        auto j = ivs[1];
        auto k = ivs[2];
        auto ld_a = nestedBuilder.create<AffineLoadOp>(
          nestedBuilder.getUnknownLoc(), A, ValueRange({i, k}));
        auto ld_b = nestedBuilder.create<AffineLoadOp>(
          nestedBuilder.getUnknownLoc(), B, ValueRange({k, j}));
        auto ld_accum = nestedBuilder.create<AffineLoadOp>(
          nestedBuilder.getUnknownLoc(), C, ValueRange({i, j}));
        if (dtype.isa<FloatType>()) {
          auto mul = nestedBuilder.create<arith::MulFOp>(
            nestedBuilder.getUnknownLoc(), ld_a, ld_b);
          auto add = nestedBuilder.create<arith::AddFOp>(
            nestedBuilder.getUnknownLoc(), mul, ld_accum);
          nestedBuilder.create<AffineStoreOp>(
            nestedBuilder.getUnknownLoc(), add, C, ValueRange({i, j}));
        }
        else {
          auto mul = nestedBuilder.create<arith::MulIOp>(
            nestedBuilder.getUnknownLoc(), ld_a, ld_b);
          auto add = nestedBuilder.create<arith::AddIOp>(
            nestedBuilder.getUnknownLoc(), mul, ld_accum);
          nestedBuilder.create<AffineStoreOp>(
            nestedBuilder.getUnknownLoc(), add, C, ValueRange({i, j}));            
        }
      }
    );
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
   });
}

std::unique_ptr<OperationPass<ModuleOp>> GEMMImplementPass() {
   return std::make_unique<GEMMImplement>();
}

struct ReluImplement : 
  public PassWrapper<ReluImplement, OperationPass<ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReluImplement)
   ReluImplement() {}
   void runOnOperation() override;
};

void ReluImplement::runOnOperation() {
   ModuleOp module = getOperation();
   if (module->hasAttr("compute_dag.relu_kernel")) {
    return;
   }
   // The walker proceeds in pre-order to process
   module.walk<WalkOrder::PreOrder>([&](compute_dag::ReluOp reluOp) {
    auto outType = reluOp.getResult().getType();
    int64_t m {-1}, n {-1};
    Type dtype;
    if(outType.isa<MemRefType>()) {
      auto outShape = outType.dyn_cast<MemRefType>();
      m = outShape.getShape()[0];
      n = outShape.getShape()[1];
      dtype = outShape.getElementType();
    }
    else {
      llvm::errs() << "Unsupported tensor type of the output of the Relu.";
      return;
    }

    OpBuilder builder(module.getContext());
    module->setAttr(std::string("compute_dag.relu_kernel"), 
      builder.getStringAttr("True"));

    builder.setInsertionPointToEnd(module.getBody());
    OpBuilder::InsertionGuard guard(builder);

    auto int32Type = builder.getI32Type();
    std::vector<Type> typesArray 
      {outType, int32Type, int32Type};
    ArrayRef<Type> paramTypes(typesArray);
    auto functionType = builder.getFunctionType(TypeRange(paramTypes), llvm::None);
    auto func = builder.create<func::FuncOp>(
      builder.getUnknownLoc(), StringRef("relu_kernel"), functionType);

    func->getRegion(0).push_back(new Block);
    Block &bodyBlock = func.front();
    int nums = static_cast<int>(paramTypes.size());
    for (int i = 0; i < nums; i++ ) {
      bodyBlock.addArguments(paramTypes[i], builder.getUnknownLoc());
    }
    ValueRange operands = bodyBlock.getArguments();
    builder.setInsertionPointToStart(&bodyBlock);  
  
    // // build loops
    SmallVector<int64_t, 2> lowerBounds(2, /*Value=*/0);
    SmallVector<int64_t, 2> steps(2, /*Value=*/1);
    SmallVector<int64_t, 2> upperBounds({m, n});
    buildAffineLoopNest(
      builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        auto tensor = operands[0];
        auto i = ivs[0];
        auto j = ivs[1];
        auto ld_element = nestedBuilder.create<AffineLoadOp>(
          nestedBuilder.getUnknownLoc(), tensor, ValueRange({i, j}));
        if (dtype.isa<FloatType>()) {
          auto zeroFloat = nestedBuilder.create<arith::ConstantFloatOp>(
            nestedBuilder.getUnknownLoc(), llvm::APFloat(0.0f), dtype.dyn_cast<FloatType>());
          auto max = nestedBuilder.create<arith::MaxFOp>(
            nestedBuilder.getUnknownLoc(), zeroFloat, ld_element);
          nestedBuilder.create<AffineStoreOp>(
            nestedBuilder.getUnknownLoc(), max, tensor, ValueRange({i, j}));
        }
        else {
          auto zeroInteger = nestedBuilder.create<arith::ConstantIntOp>(
            nestedBuilder.getUnknownLoc(), 0UL, dtype);
          auto max = nestedBuilder.create<arith::MaxSIOp>(
            nestedBuilder.getUnknownLoc(), zeroInteger, ld_element);
          nestedBuilder.create<AffineStoreOp>(
            nestedBuilder.getUnknownLoc(), max, tensor, ValueRange({i, j}));        
        }
      }
    );
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
   });
}

std::unique_ptr<OperationPass<ModuleOp>> ReluImplementPass() {
   return std::make_unique<ReluImplement>();
}

} // end namespace

namespace KernelCodegen {


void ComputeDAG::operatorImpl() {
  mlir::PassManager pm(module.getContext());
  pm.addPass(GEMMImplementPass());
  if (failed(pm.run(module))) {
    llvm::errs() << "Implement GEMM failed.";
  }
  pm.addPass(ReluImplementPass());
  if (failed(pm.run(module))) {
    llvm::errs() << "Implement Relu failed.";
  }
}

}