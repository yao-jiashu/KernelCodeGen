#include "MLIREnhance.h"
#include "utils.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>

// build a rule to implement one op
// for example, the outer loop num is related to output's shape
// the inner loop's num is related to reduction operation
// based on this rule, we can implement OperatorFuse
namespace {
using namespace mlir;
using namespace KernelCodegen;
using FuncOpMap = std::unordered_map<std::string, func::FuncOp>;
using OperatorElementMapping = 
        function_ref<void(OpBuilder &, Location, ValueRange, 
                          ValueRange, const std::vector<int> &)>;
using OperatorElementMappingPool = 
        std::unordered_map<std::string, OperatorElementMapping>;

static FuncOpMap funcCache;

// Operator Implement Pool
static OperatorElementMappingPool opiPool;

inline void flushFuncCache(ModuleOp module) {
  module.walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
    funcOp.erase();
  });
  funcCache.clear();
}


/// Creates an affine loop from the bounds known to be constants.
static AffineForOp
buildAffineLoopFromConstants(OpBuilder &builder, Location loc, int64_t lb,
                             int64_t ub, int64_t step,
                             AffineForOp::BodyBuilderFn bodyBuilderFn) {
  return builder.create<AffineForOp>(loc, lb, ub, step, /*iterArgs=*/llvm::None,
                                     bodyBuilderFn);
}

/// Creates an affine loop from the bounds that may or may not be constants.
static AffineForOp
buildAffineLoopFromValues(OpBuilder &builder, Location loc, Value lb, Value ub,
                          int64_t step,
                          AffineForOp::BodyBuilderFn bodyBuilderFn) {
  auto lbConst = lb.getDefiningOp<arith::ConstantIndexOp>();
  auto ubConst = ub.getDefiningOp<arith::ConstantIndexOp>();
  if (lbConst && ubConst)
    return buildAffineLoopFromConstants(builder, loc, lbConst.value(),
                                        ubConst.value(), step, bodyBuilderFn);
  return builder.create<AffineForOp>(loc, lb, builder.getDimIdentityMap(), ub,
                                     builder.getDimIdentityMap(), step,
                                     /*iterArgs=*/llvm::None, bodyBuilderFn);
}

/// Builds an affine loop nest, using "loopCreatorFn" to create individual loop
/// operations.
template <typename BoundListTy, typename LoopCreatorTy>
static void buildSpatialLoopNestImpl(
    OpBuilder &builder, Location loc, BoundListTy lbs, BoundListTy ubs,
    ArrayRef<int64_t> steps,
    ValueRange tensors, 
    const std::vector<int>& extraParam,
    OperatorElementMapping bodyBuilderFn,
    LoopCreatorTy &&loopCreatorFn) {
  assert(lbs.size() == ubs.size() && "Mismatch in number of arguments");
  assert(lbs.size() == steps.size() && "Mismatch in number of arguments");

  // If there are no loops to be constructed, construct the body anyway.
  OpBuilder::InsertionGuard guard(builder);
  if (lbs.empty()) {
    if (bodyBuilderFn)
      bodyBuilderFn(builder, loc, ValueRange(), tensors, extraParam);
    return;
  }

  // Create the loops iteratively and store the induction variables.
  SmallVector<Value, 4> ivs;
  ivs.reserve(lbs.size());
  for (unsigned i = 0, e = lbs.size(); i < e; ++i) {
    // Callback for creating the loop body, always creates the terminator.
    auto loopBody = [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
                        ValueRange iterArgs) {
      ivs.push_back(iv);
      // In the innermost loop, call the body builder.
      if (i == e - 1 && bodyBuilderFn) {
        OpBuilder::InsertionGuard nestedGuard(nestedBuilder);
        bodyBuilderFn(nestedBuilder, nestedLoc, ivs, tensors, extraParam);
      }
      nestedBuilder.create<AffineYieldOp>(nestedLoc);
    };

    // Delegate actual loop creation to the callback in order to dispatch
    // between constant- and variable-bound loops.
    auto loop = loopCreatorFn(builder, loc, lbs[i], ubs[i], steps[i], loopBody);
    builder.setInsertionPointToStart(loop.getBody());
  }
}

void buildSpatialLoopNest(
    OpBuilder &builder, Location loc, ValueRange lbs, ValueRange ubs,
    ArrayRef<int64_t> steps,
    ValueRange tensors, 
    const std::vector<int>& extraParam,
    OperatorElementMapping bodyBuilderFn) {
  buildSpatialLoopNestImpl(builder, loc, lbs, ubs, steps, tensors, extraParam, bodyBuilderFn,
                          buildAffineLoopFromValues);
}


///TODO:wrapper this using lambda function
// Extend operators here, Rule:
// tensors meet the order: output, lhs. rhs,...
// ivs' size is equals to output's dimension's size, 
//      and ivs[i] represent the iteration var of  output.shape[i] 
// all requirement is highly consistent with operations in ComputeDAGDialect
void operatorElementMapping() {
  // GEMMOp
  opiPool["gemm"] = [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs, 
          ValueRange tensors, const std::vector<int>& extraParam) {
    auto C = tensors[0];
    auto A = tensors[1];
    auto B = tensors[2];
    auto i = ivs[0];
    auto j = ivs[1];

    int K = extraParam[0];
    auto CType = C.getType();
    auto dtype = CType.dyn_cast<MemRefType>().getElementType();

    auto ld_c = nestedBuilder.create<AffineLoadOp>(
          nestedBuilder.getUnknownLoc(), C, ValueRange({i, j}));

    auto kLoopBody = [&](OpBuilder &builder, Location nestedLoc, Value iv,
                        ValueRange iterArgs) {
      OpBuilder::InsertionGuard nestedGuard(builder);
      auto c_element = iterArgs[0];
      auto k = iv;
      auto ld_a = builder.create<AffineLoadOp>(
                    builder.getUnknownLoc(), A, ValueRange({i, k}));
      auto ld_b = builder.create<AffineLoadOp>(
                    builder.getUnknownLoc(), B, ValueRange({k, j}));
      if (dtype.isa<FloatType>()) {
        auto mul = builder.create<arith::MulFOp>(
                    builder.getUnknownLoc(), ld_a, ld_b);
        auto add = builder.create<arith::AddFOp>(
                    builder.getUnknownLoc(), mul, c_element);
        builder.create<AffineYieldOp>(builder.getUnknownLoc(), add.getResult());
      }
      else {
        auto mul = builder.create<arith::MulIOp>(
          builder.getUnknownLoc(), ld_a, ld_b);
        auto add = builder.create<arith::AddIOp>(
          builder.getUnknownLoc(), mul, c_element);
        builder.create<AffineYieldOp>(builder.getUnknownLoc(), add.getResult());
      }
    };

    auto loopCarriedVar = ld_c.getResult();
    auto forOp = nestedBuilder.create<AffineForOp>(nestedBuilder.getUnknownLoc(), 
                    0, K, 1, /*iterArgs=lvm::None*/ ValueRange({loopCarriedVar}), kLoopBody);
        // module->setAttr(std::string("compute_dag.gemm_kernel"), 
    //   builder.getStringAttr("True"));
    forOp->setAttr(std::string("compute_dag.loop_attr"),
        nestedBuilder.getStringAttr("reduction"));
    nestedBuilder.create<AffineStoreOp>(
        nestedBuilder.getUnknownLoc(), forOp.getResult(0), C, ValueRange({i, j}));
  };

  // ReluOp
  opiPool["relu"] = [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs, 
          ValueRange tensors, const std::vector<int>& extraParam) {
    auto tensor = tensors[0];
    auto i = ivs[0];
    auto j = ivs[1];

    auto tensorType = tensor.getType();
    auto dtype = tensorType.dyn_cast<MemRefType>().getElementType();

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
  };

}


struct GEMMImplement : 
  public PassWrapper<GEMMImplement, OperationPass<ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GEMMImplement)
   GEMMImplement() {}
   void runOnOperation() override;
};

void GEMMImplement::runOnOperation() {
   ModuleOp module = getOperation();
  //  if (module->hasAttr("compute_dag.gemm_kernel")) {
  //   return;
  //  }
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
    auto funcName = GEMMName(m, n, k) + "_kernel";
    if (funcCache.find(funcName) != funcCache.end())
      return;
    OpBuilder builder(module.getContext());
    // module->setAttr(std::string("compute_dag.gemm_kernel"), 
    //   builder.getStringAttr("True"));

    builder.setInsertionPointToEnd(module.getBody());
    OpBuilder::InsertionGuard guard(builder);

    auto int32Type = builder.getI32Type();
    std::vector<Type> typesArray 
      {outType, typeA, typeB, int32Type, int32Type, int32Type};
    ArrayRef<Type> paramTypes(typesArray);
    auto functionType = builder.getFunctionType(TypeRange(paramTypes), llvm::None);

    auto func = builder.create<func::FuncOp>(
      builder.getUnknownLoc(), StringRef(funcName), functionType);

    funcCache[funcName] = func;

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

    auto funcName = ReluName(m, n) + "_kernel";
    if (funcCache.find(funcName) != funcCache.end()) {
      return;
    }
    OpBuilder builder(module.getContext());

    builder.setInsertionPointToEnd(module.getBody());
    OpBuilder::InsertionGuard guard(builder);

    auto int32Type = builder.getI32Type();
    std::vector<Type> typesArray 
      {outType, int32Type, int32Type};
    ArrayRef<Type> paramTypes(typesArray);
    auto functionType = builder.getFunctionType(TypeRange(paramTypes), llvm::None);
    auto func = builder.create<func::FuncOp>(
      builder.getUnknownLoc(), StringRef(funcName), functionType);
    
    funcCache[funcName] = func;

    func->getRegion(0).push_back(new Block);
    Block &bodyBlock = func.front();
    int nums = static_cast<int>(paramTypes.size());
    for (int i = 0; i < nums; i++ ) {
      bodyBlock.addArguments(paramTypes[i], builder.getUnknownLoc());
    }
    ValueRange operands = bodyBlock.getArguments();
    builder.setInsertionPointToStart(&bodyBlock);  
  
    // build loops
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

struct OperatorImplement : 
  public PassWrapper<OperatorImplement, OperationPass<ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OperatorImplement)
   OperatorImplement() {}
   void runOnOperation() override;
};

void OperatorImplement::runOnOperation() {
  ModuleOp module = this->getOperation();

  auto& operations = module.getBody()->getOperations();

  std::vector<std::vector<Operation*>> kernelList;
  std::vector<Operation*> kernel;
  for (auto iter = operations.begin(); iter != operations.end(); iter++) {
    auto& op = *iter;
    if (op.getDialect()->getNamespace() == "compute_dag") {
      if (isa<compute_dag::DelimiterOp>(op)) {
        // auto delimiterOp = dyn_cast<compute_dag::DelimiterOp>(op);
        assert(kernel.size() != 0);
        kernelList.push_back(kernel);
        kernel.clear();
      }
      else {
        kernel.push_back(&op);
      }
    }
  }
  for (auto& kernel : kernelList) {
    MemRefType memoryType;
    std::vector<int> extraParams;
    if (isa<compute_dag::GEMMOp>(*(kernel[0]))) {
      auto gemmOp = dyn_cast<compute_dag::GEMMOp>(*(kernel[0]));
      memoryType = gemmOp.getResult().getType().dyn_cast<MemRefType>();
      int k = gemmOp.getOperands()[0].getType().dyn_cast<MemRefType>().getShape()[1];
      extraParams.push_back(k);
    }
    else if (isa<compute_dag::ReluOp>(*(kernel[0]))) {
      auto reluOp = dyn_cast<compute_dag::ReluOp>(*(kernel[0]));
      memoryType = reluOp.getResult().getType().dyn_cast<MemRefType>();
    }
    int m = memoryType.getShape()[0];
    int n = memoryType.getShape()[1];

    OpBuilder builder(module.getContext());

    SmallVector<int64_t, 2> lowerBounds(2, /*Value=*/0);
    SmallVector<int64_t, 2> steps(2, /*Value=*/1);
    SmallVector<int64_t, 2> upperBounds({m, n});

    buildSpatialLoopNest(builder, builder.getUnknownLoc(), 
      lowerBounds, upperBounds, steps,
      
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs, 
          ValueRange tensors, const std::vector<int>& extraParam) {
      }
      OperatorElementMapping bodyBuilderFn, 
      ValueRange tensors, 
      const std::vector<int>& extraParam)
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

    auto funcName = ReluName(m, n) + "_kernel";
    if (funcCache.find(funcName) != funcCache.end()) {
      return;
    }
    OpBuilder builder(module.getContext());

    builder.setInsertionPointToEnd(module.getBody());
    OpBuilder::InsertionGuard guard(builder);

    auto int32Type = builder.getI32Type();
    std::vector<Type> typesArray 
      {outType, int32Type, int32Type};
    ArrayRef<Type> paramTypes(typesArray);
    auto functionType = builder.getFunctionType(TypeRange(paramTypes), llvm::None);
    auto func = builder.create<func::FuncOp>(
      builder.getUnknownLoc(), StringRef(funcName), functionType);
    
    funcCache[funcName] = func;

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

std::unique_ptr<OperationPass<ModuleOp>> OperatorImplementPass() {
   return std::make_unique<OperatorImplement>();
}

} // end namespace

namespace KernelCodegen {

void ComputeDAG::registerElementMapping() {
  operatorElementMapping();
}

void ComputeDAG::operatorImpl() {
  flushFuncCache(module);
  // mlir::PassManager pm(module.getContext());
  // pm.addPass(GEMMImplementPass());
  // if (failed(pm.run(module))) {
  //   llvm::errs() << "Implement GEMM failed.";
  // }
  // pm.addPass(ReluImplementPass());
  // if (failed(pm.run(module))) {
  //   llvm::errs() << "Implement Relu failed.";
  // }
  mlir::PassManager pm(module.getContext());
  pm.addPass(OperatorImplementPass());
  if (failed(pm.run(module))) {
    llvm::errs() << "Implement Operators failed.";
  }
}

}