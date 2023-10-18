#include "Scheduler.h"
#include "utils.h"

// used to sample static information

using namespace mlir;
using namespace KernelCodegen;
namespace {

//----------------------------------------------- collect functions---------------------------------//

static std::vector<Scheduler::Function> funcs;

struct CollectFuncOp : 
  public PassWrapper<CollectFuncOp, OperationPass<ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CollectFuncOp)
   CollectFuncOp(std::string& name_, ConstPassGuard* passGuard_) : name(name_), passGuard(passGuard_) {}
   void runOnOperation() override;
   std::string name;
   ConstPassGuard* passGuard;
};

void CollectFuncOp::runOnOperation() {
  ModuleOp module = getOperation();
  
  if (passGuard->visited()) return;
  passGuard->visit();

  module.walk<WalkOrder::PreOrder>([&](Scheduler::Function funcOp) {
    // get the name of the func::FuncOp
    std::string funcName {funcOp.getSymName()};
    if (funcName.find(name) != std::string::npos) {
      funcs.push_back(funcOp);
    }
  });
}

std::unique_ptr<OperationPass<ModuleOp>> CollectFunctionsPass(
  std::string& name, ConstPassGuard* passGuard) {
  return std::make_unique<CollectFuncOp>(name, passGuard);
}

//----------------------------------------------- collect loops---------------------------------//

static std::vector<LoopInfo> loopInfos;

void DFS(Operation* op, int scope) {
  LoopInfo loopInfo;
  loopInfo.scope = scope;
  if (isa<AffineForOp>(*op)) {
    scope += 1;
    auto forOp = dyn_cast<AffineForOp>(*op);
    loopInfo.forOp = forOp;
    OpBuilder builder(forOp.getContext());

    auto attr = forOp->getAttr(builder.getStringAttr("schedule.loop_attr"));
    if (attr != nullptr) {
      auto attrStr =attr.dyn_cast<StringAttr>().str();
      if (attrStr == "spatial") loopInfo.attibute = LoopAttribute::spatial;
      else if (attrStr == "reduction") loopInfo.attibute = LoopAttribute::reduction;
      else assert(false);
    }
    else {
      assert(false);
    }
    loopInfos.emplace_back(loopInfo);
  }
  int numRegion = op->getNumRegions();
  if (numRegion != 0) {
    auto regions = op->getRegions();
    for (auto& region : regions) {
      auto& blocks = region.getBlocks();
      for (auto& block : blocks) {
        auto& ops = block.getOperations();
        for (auto& op : ops) {
          DFS(&op, scope);
        }
      }
    }
  }
}

struct CollectAffineForOp : 
  public PassWrapper<CollectAffineForOp, OperationPass<ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CollectAffineForOp)
   CollectAffineForOp(Scheduler::Function& func_, ConstPassGuard* passGuard_) : func(func_), passGuard(passGuard_) {}
   void runOnOperation() override;
   Scheduler::Function func;
   ConstPassGuard* passGuard;
};

void CollectAffineForOp::runOnOperation() {
  ModuleOp module = getOperation();
  
  if (passGuard->visited()) return;
  passGuard->visit();

  module.walk<WalkOrder::PreOrder>([&](Scheduler::Function funcOp) {

    if (funcOp != func) {
      return;
    }
    // depth-first search
    DFS(funcOp, 0);
  });

}

std::unique_ptr<OperationPass<ModuleOp>> CollectAffineForOpPass(
  Scheduler::Function& func, ConstPassGuard* passGuard) {
  
  return std::make_unique<CollectAffineForOp>(func, passGuard);
}


//----------------------------------------------- load store ---------------------------------//

/// lowering affine load/store to memref load/store
struct LoweringAffineLoadStore
 : public PassWrapper<LoweringAffineLoadStore, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoweringAffineLoadStore)
   LoweringAffineLoadStore() {}
   void runOnOperation() override;
};
 
/// Apply the affine map from an 'affine.load' operation to its operands, and
/// feed the results to a newly created 'memref.load' operation (which replaces
/// the original 'affine.load').
class AffineLoadLowering : public OpRewritePattern<AffineLoadOp> {
public:
  using OpRewritePattern<AffineLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineLoadOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineLoadOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto resultOperands =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!resultOperands)
      return failure();

    // Build vector.load memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, op.getMemRef(),
                                                *resultOperands);
    return success();
  }
};


/// Apply the affine map from an 'affine.store' operation to its operands, and
/// feed the results to a newly created 'memref.store' operation (which replaces
/// the original 'affine.store').
class AffineStoreLowering : public OpRewritePattern<AffineStoreOp> {
public:
  using OpRewritePattern<AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineStoreOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineStoreOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    // Build memref.store valueToStore, memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, op.getValueToStore(), op.getMemRef(), *maybeExpandedMap);
    return success();
  }
};

void LoweringAffineLoadStore::runOnOperation() {
    RewritePatternSet patterns(&getContext());
  // clang-format off
    patterns.add<
      AffineLoadLowering,
      AffineStoreLowering
      >(patterns.getContext());
  // clang-format on
    ConversionTarget target(getContext());
    target.addLegalOp<AffineYieldOp, AffineForOp>();
    target.addLegalDialect<arith::ArithmeticDialect, memref::MemRefDialect,
                           scf::SCFDialect, vector::VectorDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLoweringAffinePass() {
  return std::make_unique<LoweringAffineLoadStore>();
}


static std::vector<Value> insAndOuts;

struct CollectInsAndOuts : 
  public PassWrapper<CollectInsAndOuts, OperationPass<func::FuncOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CollectInsAndOuts)
   CollectInsAndOuts(ConstPassGuard* passGuard_) : passGuard(passGuard_) {}
   void runOnOperation() override;
   ConstPassGuard* passGuard;
};

void CollectInsAndOuts::runOnOperation() {
  if (passGuard->visited()) return;
  passGuard->visit();
  func::FuncOp func = getOperation();

  auto args = func.getArguments();
  for (auto& arg : args) {
    auto type = arg.getType();
    if(type.isa<MemRefType>()) {
      insAndOuts.push_back(arg);
    }
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> CollectInsAndOutsPass(
  ConstPassGuard* passGuard) {
   return std::make_unique<CollectInsAndOuts>(passGuard);
}

}

namespace KernelCodegen {

std::vector<Scheduler::Function> 
Scheduler::collectFunctions(std::string&& functionName) {
  ConstPassGuard passGuard;
  PassManager pm(graph->module.getContext());
 
  funcs.clear();
  
  pm.addPass(CollectFunctionsPass(functionName, &passGuard));

  if (failed(pm.run(graph->module))) {
    llvm::errs() << "Collects functions failed.\n";
  }

  return funcs;
}

/// @brief TODO: optimize this function
/// @return 
std::vector<LoopInfo> Scheduler::collectLoops(Scheduler::Function& func) {
  loopInfos.clear();
  ConstPassGuard passGuard;
  PassManager pm(graph->module.getContext());
  pm.addPass(CollectAffineForOpPass(func, &passGuard));
  if (failed(pm.run(graph->module))) {
    llvm::errs() << "Collects loops information failed.\n";
  }
  return loopInfos;
}

void Scheduler::loweringAffineLoadStore() {
  PassManager pm(graph->module.getContext());
  OpPassManager &optPM = pm.nest<func::FuncOp>();
  optPM.addPass(createLoweringAffinePass());
  if (failed(pm.run(graph->module))) {
    llvm::errs() << "Lowering Affine Load/Store Op failed.\n";
  }
}

std::vector<Value> Scheduler::collectInputsAndOutputs() {
  insAndOuts.clear();
  ConstPassGuard passGuard;
  PassManager pm(graph->module.getContext());
  OpPassManager &optPM = pm.nest<func::FuncOp>();
  optPM.addPass(CollectInsAndOutsPass(&passGuard));
  if (failed(pm.run(graph->module))) {
    llvm::errs() << "Collects inputs and outpus information failed.\n";
  }
  return insAndOuts;
}

Scheduler::DType Scheduler::getDataType(std::string dtype) {
  if (dtype == "float32") {
    return graph->builder.getF32Type();
  }
  return nullptr;
}

/// Creates an loop from the bounds known to be constants.
Scheduler::Loop buildLoop(OpBuilder &builder, Location loc, int64_t lb, int64_t ub, int64_t step,
                          Scheduler::Loop::BodyBuilderFn bodyBuilderFn) {
  return builder.create<Scheduler::Loop>(loc, lb, ub, step, /*iterArgs=*/llvm::None, bodyBuilderFn);
}


/// Builds an affine loop nest, using "loopCreatorFn" to create individual loop
/// operations.
template <typename BoundListTy, typename LoopCreatorTy>
static void buildLoopNestImpl(
    OpBuilder &builder, Location loc, BoundListTy lbs, BoundListTy ubs,
    ArrayRef<int64_t> steps,
    Scheduler::LoopBuildFn bodyBuilderFn,
    LoopCreatorTy &&loopCreatorFn) {
  assert(lbs.size() == ubs.size() && "Mismatch in number of arguments");
  assert(lbs.size() == steps.size() && "Mismatch in number of arguments");

  // If there are no loops to be constructed, construct the body anyway.
  OpBuilder::InsertionGuard guard(builder);
  if (lbs.empty()) {
    if (bodyBuilderFn)
      bodyBuilderFn(builder, loc, ValueRange());
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
        bodyBuilderFn(nestedBuilder, nestedLoc, ivs);
      }
      nestedBuilder.create<AffineYieldOp>(nestedLoc);
    };

    // Delegate actual loop creation to the callback in order to dispatch
    // between constant- and variable-bound loops.
    auto loop = loopCreatorFn(builder, loc, lbs[i], ubs[i], steps[i], loopBody);
    builder.setInsertionPointToStart(loop.getBody());
  }
}

void Scheduler::buildLoopNest(
    OpBuilder &builder, Location loc, 
    ArrayRef<int64_t> lbs,
    ArrayRef<int64_t> ubs, 
    ArrayRef<int64_t> steps,
    Scheduler::LoopBuildFn bodyBuilderFn) {
  buildLoopNestImpl(builder, loc, lbs, ubs, steps, bodyBuilderFn, buildLoop);
}

std::vector<Value> Scheduler::createOpsFromExpressions(std::vector<Tensor::Expr>& exprs, OpBuilder& builder) {

  auto getValueFromOperand = [&](Operand& opr) -> Value {
    if (opr.type == Operand::OperandType::Value) {
      return opr.value;
    } else if (opr.type == Operand::OperandType::Integer) {
      auto integer = builder.create<arith::ConstantIndexOp>(
          builder.getUnknownLoc(), opr.integer);
      return integer.getResult();
    } else {
      llvm::errs() << "Not support this Operand type in Expression\n";
    }
  };

  std::vector<Value> res;
  for (auto& expr : exprs) {
    auto left = getValueFromOperand(expr->left);
    auto right = getValueFromOperand(expr->right);

    if (expr->op == Operator::Add) {
      auto exprOp = builder.create<arith::AddIOp>(builder.getUnknownLoc(), left, right);
      res.push_back(exprOp.getResult());
    } else if (expr->op == Operator::Mul) {
      auto exprOp = builder.create<arith::MulIOp>(builder.getUnknownLoc(), left, right);
      res.push_back(exprOp.getResult());
    } else {
      llvm::errs() << "Not support this Operator in Expression\n";
    }
  }

  return res;

}

}