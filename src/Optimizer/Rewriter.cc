#include "Optimizer/Rewriter.h"
#include "enum.h"

#include "llvm/ADT/ArrayRef.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "mlir/IR/PatternMatch.h"

#include <algorithm>
#include <map>
#include <cmath>

namespace KernelCodeGen {

mlir::AffineExpr shiftAffineExprDim(mlir::MLIRContext* context, mlir::AffineExpr expr, int shift) {
  if (auto dimExpr_ = expr.dyn_cast<mlir::AffineDimExpr>()) {
    return mlir::getAffineDimExpr(dimExpr_.getPosition() + shift, context);
  } else if (auto binaryExpr_ = expr.dyn_cast<mlir::AffineBinaryOpExpr>()){
    auto LHS = shiftAffineExprDim(context, binaryExpr_.getLHS(), shift);
    auto RHS = shiftAffineExprDim(context, binaryExpr_.getRHS(), shift);
    return mlir::getAffineBinaryOpExpr(binaryExpr_.getKind(), LHS, RHS);
  } else {
    // allowed dim, constant, binaryOp
    auto constExpr_ = expr.dyn_cast<mlir::AffineConstantExpr>();
    assert(constExpr_);
    return constExpr_;
  }
}

mlir::AffineExpr getModifiedExpr(mlir::MLIRContext* context, mlir::AffineExpr inExpr, mlir::AffineExpr replaceExpr, int targetDim, int replaceNumberDims) {
  if (auto dimExpr_ = inExpr.dyn_cast<mlir::AffineDimExpr>()) {
    if (dimExpr_.getPosition() == targetDim) {
      return replaceExpr;
    } else if (dimExpr_.getPosition() > targetDim) {
      return mlir::getAffineDimExpr(dimExpr_.getPosition() + replaceNumberDims - 1, context);
      // return dimExpr_.shiftDims(1, replaceNumberDims - 1, /*offset*/0);
    } else {
      return dimExpr_;
    }
  } else if (auto binaryExpr_ = inExpr.dyn_cast<mlir::AffineBinaryOpExpr>()){
    auto LHS = getModifiedExpr(context, binaryExpr_.getLHS(), replaceExpr, targetDim, replaceNumberDims);
    auto RHS = getModifiedExpr(context, binaryExpr_.getRHS(), replaceExpr, targetDim, replaceNumberDims);
    return mlir::getAffineBinaryOpExpr(binaryExpr_.getKind(), LHS, RHS);
  } else {
    // allowed dim, constant, binaryOp
    auto constExpr_ = inExpr.dyn_cast<mlir::AffineConstantExpr>();
    assert(constExpr_);
    return constExpr_;
  }
}

mlir::AffineForOp findRootLoop(mlir::Operation* op) {
  while (true) {
    auto parentOp = op->getParentOp();
    if (!parentOp) assert(false);
    if (auto module = mlir::dyn_cast<mlir::ModuleOp>(parentOp)) {
      return mlir::dyn_cast<mlir::AffineForOp>(op);
    } else if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(parentOp)){
      return mlir::dyn_cast<mlir::AffineForOp>(op);
    } else if (auto parallel = mlir::dyn_cast<mlir::AffineParallelOp>(parentOp)) {
      return mlir::dyn_cast<mlir::AffineForOp>(op);
    }
    op = mlir::dyn_cast<mlir::AffineForOp>(parentOp);
    if (!op) {
      op = mlir::dyn_cast<mlir::AffineIfOp>(parentOp);
    }
    if (!op) {
      assert(false);
    }
  }
}


template <typename AffineMemoryOp>
int replaceIndexWithExpr(mlir::Value oldIv, std::vector<mlir::Value>& newIvs, AffineMemoryOp memOp, mlir::AffineExpr replaceExpr,
                         llvm::SmallVector<mlir::AffineExpr>& exprs, llvm::SmallVector<mlir::Value>& operands) {
  mlir::OpBuilder builder(memOp);

  llvm::SmallVector<mlir::Value> operands_(memOp.getMapOperands());
  int targetDim = -1;
  bool found = false;
  // d0, d1, d2, [d3], d4 ->(d3)->d0, d1, d2, [d3, d4, d5], d6
  for (auto item : operands_) {
    if (!found) targetDim += 1;
    if (item == oldIv) {
      found = true;
      for (auto iv : newIvs) { operands.push_back(iv); }
      // break;
    } else {
      operands.push_back(item);
    }
  }
  assert(found);

  ///Debug info
  // llvm::errs() << "operands size " << operands.size() << "\n";
  // llvm::errs() << "targetDim  " << targetDim<< "\n";
  replaceExpr = shiftAffineExprDim(builder.getContext(), replaceExpr, targetDim);
  // llvm::errs() << "replace dump: ";
  // replaceExpr.dump();

  auto map = memOp.getAffineMap();
  auto exprs_ = map.getResults();

  for (auto expr_ : exprs_) {
    auto expr = getModifiedExpr(builder.getContext(), expr_, replaceExpr, targetDim, newIvs.size());
    // expr.dump();
    exprs.push_back(expr);
  }  
  return operands.size();
}

std::vector<mlir::AffineForOp> Rewriter::split(mlir::AffineForOp forOp,
                                            uint64_t num_output, std::vector<int64_t>&& factors) {
  auto upperBoundsVector = factors;
  factors.insert(factors.begin(), 1);
  assert(factors.size() == num_output);
  std::reverse(factors.begin(), factors.end());

  auto lowerbound = forOp.getLowerBoundMap();
  auto upperbound = forOp.getUpperBoundMap();
  int step = forOp.getStep();
  assert(lowerbound.isConstant() == true);
  assert(upperbound.isConstant() == true);
  int64_t lb = lowerbound.getSingleConstantResult();
  int64_t ub = upperbound.getSingleConstantResult();
  assert(step == 1 && lb == 0);

  upperBoundsVector.push_back(ub);
  std::reverse(upperBoundsVector.begin(), upperBoundsVector.end());

  mlir::SmallVector<int64_t, 16> lowerBounds(num_output, /*Value=*/0);
  mlir::SmallVector<int64_t, 16> steps(factors.begin(), factors.end());
  mlir::SmallVector<int64_t, 16> upperBounds(upperBoundsVector.begin(), upperBoundsVector.end());

  std::vector<mlir::Value> ivsVector;

  mlir::OpBuilder builder(forOp.getOperation());
  mlir::buildAffineLoopNest(
    builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
      //empty nested loops.
      for (auto iv : ivs) {
        ivsVector.push_back(iv);
      }
    }
  );

  // build AffineMap: (i) -> (i1 + i2 + i3)

  auto prevNode = forOp->getPrevNode();
  std::vector<mlir::AffineForOp> loops;
  mlir::AffineForOp outermostForOp = mlir::dyn_cast<mlir::AffineForOp>(prevNode);
  outermostForOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineForOp newLoop) {
    loops.push_back(newLoop);
  });

  mlir::AffineForOp innermostForOp = loops.back();
  // erase the yield op, as the forOp will bring the AffineYieldOp
  innermostForOp.getBody()->back().erase();
  innermostForOp.getBody()->getOperations().splice(innermostForOp.getBody()->end(),
                                                    forOp.getBody()->getOperations());

  /* Method 1: Replace old iv with new iv attached affineMapAttr */

  auto oldIv = forOp.getInductionVar();
  auto users = oldIv.getUsers();

  int dimCount = 0;
  ///TODO: can be passed as a functional<>, so free the build style of expr.
  std::vector<mlir::AffineExpr> dims;
  mlir::AffineExpr sumExpr;
  for (int i = 0; i < ivsVector.size(); i += 1 ) {
    dims.push_back(std::move(builder.getAffineDimExpr(dimCount++)));
    if (i == 0) {
      sumExpr = dims[0];
    } else {
      sumExpr = sumExpr + dims.back();
    }
  }

  for (auto user : users) {
    mlir::OpBuilder builder(user);
    if (auto loadOp = mlir::dyn_cast<mlir::AffineLoadOp>(user)) {
      llvm::SmallVector<mlir::AffineExpr> exprs;
      llvm::SmallVector<mlir::Value> operands;
      int dimCount = replaceIndexWithExpr<mlir::AffineLoadOp>(oldIv, ivsVector, loadOp, sumExpr, exprs, operands);
      mlir::AffineMap map = mlir::AffineMap::get(/*dimCount*/dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
      auto mem = loadOp.getMemref();
      auto newLoadOp = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), mem, map, llvm::ArrayRef<mlir::Value>(operands));
      loadOp.getResult().replaceAllUsesWith(newLoadOp.getResult());
      loadOp.erase();
    } else if (auto storeOp = mlir::dyn_cast<mlir::AffineStoreOp>(user)) {
      llvm::SmallVector<mlir::AffineExpr> exprs;
      llvm::SmallVector<mlir::Value> operands;
      auto valueToStore = storeOp.getValue();
      auto mem = storeOp.getMemref();
      int dimCount = replaceIndexWithExpr<mlir::AffineStoreOp>(oldIv, ivsVector, storeOp, sumExpr, exprs, operands);
      // llvm::errs() << exprs.size() << "滑天下之大稽\n";
      mlir::AffineMap map = mlir::AffineMap::get(/*dimCount*/dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
      builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), valueToStore, mem, map, llvm::ArrayRef<mlir::Value>(operands));
      storeOp.erase();
    } else {
      assert(false);
    }
  }

  /* Method 2: Replace old iv with AffineApplyOp's result */


  // std::vector<mlir::AffineExpr> dims;
  // mlir::AffineExpr sumExpr;
  // for (int i = 0; i < num_output; i += 1 ) {
  //   dims.push_back(std::move(builder.getAffineDimExpr(i)));
  //   if (i == 0) {
  //     sumExpr = dims[0];
  //   } else {
  //     sumExpr = sumExpr + dims.back();
  //   }
  // }
  // auto sumMap = mlir::AffineMap::get(/*dimCount*/num_output, 0, sumExpr);
  // auto attribute = mlir::AffineMapAttr::get(sumMap);
  // builder.setInsertionPointToStart(innermostForOp.getBody());
  // mlir::IRRewriter rewriter(builder);
  // // rewriter.setInsertionPointToStart();
  // auto results = mlir::getAsOpFoldResult(llvm::SmallVector<mlir::Value>(ivsVector.begin(), ivsVector.end()));
  // auto operands = llvm::ArrayRef<mlir::OpFoldResult>(llvm::SmallVector<mlir::OpFoldResult>(results));
  // auto ivReplacement = mlir::makeComposedFoldedAffineApply(rewriter, rewriter.getUnknownLoc(), sumMap, operands);
  // auto oldIv = forOp.getInductionVar();
  // oldIv.replaceAllUsesWith(ivReplacement.get<mlir::Value>());

  forOp.erase();

  return loops;
}

// mlir::Value Rewriter::bufferizeLoopCarryVar(mlir::AffineForOp loop) {

// }

mlir::Block* getClostScopeOp(mlir::Operation* op) {
  while (true) {
    auto parentOp = op->getParentOp();
    if (auto module = mlir::dyn_cast<mlir::ModuleOp>(parentOp)) {
      return module.getBody();
    } else if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(parentOp)){
      return &(func.getBlocks().front());
    } else if (auto parallelOp = mlir::dyn_cast<mlir::AffineParallelOp>(parentOp)) {
      return parallelOp.getBody();
    }
    op = parentOp;
  }
}

mlir::Value Rewriter::bufferizeLoopCarryVar(std::vector<mlir::AffineForOp>& loops) {
  auto contain = [&](mlir::AffineForOp A, mlir::AffineForOp B) {
    if (A == B) return false;
    bool result = false;
    A.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineForOp forOp) {
      if (forOp == B) {
        result = true;
      }
    });
    return result;
  };

  bool hasLoopCarryVar = false;
  mlir::AffineForOp carryVarLoop;
  std::vector<int64_t> bufferShape;
  llvm::SmallVector<mlir::Value> bufferAdrressOperand;
  int replaceIdx = -1;
  for (auto loop : loops) {
    if (!hasLoopCarryVar) replaceIdx += 1;
    auto args = loop.getRegionIterArgs();
    if (args.size() != 0) {
      if (!hasLoopCarryVar) {
        hasLoopCarryVar = true;
        carryVarLoop = loop;
        continue;
      } else {
        llvm::errs() << "Can't reorder more than one loops carrying args\n";
        assert(false);
      }
    }
    if (hasLoopCarryVar && contain(loop, carryVarLoop)) {
      int64_t ub = loop.getConstantUpperBound();
      int64_t lb = loop.getConstantLowerBound();
      int64_t step = loop.getStep();

      bufferShape.push_back((ub - lb) / step);
      bufferAdrressOperand.push_back(loop.getInductionVar());
    }
  }

  if (!hasLoopCarryVar) return nullptr;

  auto topLevelBlock = getClostScopeOp(loops[0]);

  auto builder = mlir::OpBuilder::atBlockBegin(topLevelBlock);
  auto carryVar = carryVarLoop.getRegionIterArgs()[0];
  auto dtype = carryVar.getType();
  auto bufferType = mlir::MemRefType::get(
    bufferShape, dtype, {}, static_cast<int>(MemorySpace::local));
  auto allocOp = builder.create<mlir::memref::AllocOp>(
    builder.getUnknownLoc(), bufferType);
  
  // step1: init the buffer
  // the last operand of AffineForOp.
  auto initValue = carryVarLoop.getOperands().back();
  auto defineOp = initValue.getDefiningOp();
  // init after defineOp
  builder.setInsertionPointAfter(defineOp);
  // mlir::OpBuilder builder(&*(++mlir::Block::iterator(defineOp)));
  builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), initValue, allocOp.getResult(), bufferAdrressOperand);


  // step2: replace the loop carry var
  int64_t ub = carryVarLoop.getConstantUpperBound();
  int64_t lb = carryVarLoop.getConstantLowerBound();
  int64_t step = carryVarLoop.getStep();
  builder.setInsertionPointAfter(carryVarLoop);
  mlir::Value replaceValue;
  auto loopBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv,
                      mlir::ValueRange iterArgs) {
    mlir::OpBuilder::InsertionGuard nestedGuard(builder);
    auto loadOp = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), allocOp.getResult(), bufferAdrressOperand);
    replaceValue = loadOp.getResult();
    builder.create<mlir::AffineYieldOp>(builder.getUnknownLoc());
  };
  auto newLoop = builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), lb, ub, step, 
                    /*iterArgs=lvm::None*/ mlir::ValueRange({}), loopBody);
  auto& oldYieldOp = carryVarLoop.getBody()->getOperations().back();
  // insert after loadOp;
  newLoop.getBody()->getOperations().splice(++(newLoop.getBody()->getOperations().begin()),
      carryVarLoop.getBody()->getOperations());

  carryVarLoop.getInductionVar().replaceAllUsesWith(newLoop.getInductionVar());

  carryVar.replaceAllUsesWith(replaceValue);

  // remove the yield op with loopCarryVar.
  auto yieldResult = mlir::dyn_cast<mlir::AffineYieldOp>(oldYieldOp).getOperand(0);

  builder.setInsertionPointAfter(&oldYieldOp);
  builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), yieldResult, allocOp.getResult(), bufferAdrressOperand); 

  oldYieldOp.erase();

  // step3: replace all uses of carryVarLoop's result;
  builder.setInsertionPointAfter(newLoop);
  auto loadOp = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), allocOp.getResult(), bufferAdrressOperand);
  auto carryVarLoopResult = carryVarLoop.getResult(0);
  carryVarLoopResult.replaceAllUsesWith(loadOp.getResult());

  auto users = carryVarLoopResult.getUsers();

  carryVarLoop.erase();

  loops[replaceIdx] = newLoop;

  return allocOp.getResult();
}
 
// Swap two nested loops.
// if outer loop contains multiple Operations, clone the outer loop to maintain correctness.
void swap(mlir::AffineForOp outer, mlir::AffineForOp inner) {
  auto& ops = outer.getBody()->getOperations();
  auto opNumber = ops.size();
  int position = 0;
  mlir::Operation* innerOp = inner;
  for (auto& op : ops) {
    if (&op == innerOp) {
      break;
    }
    position += 1;
  }
  // must found.
  assert(position < opNumber);

  bool existOpBeforeLoop = position != 0;
  // considering the affine.yield
  bool existOpAfterLoop = position != opNumber - 2;

  if (existOpBeforeLoop) {
    mlir::OpBuilder b(outer->getBlock(), mlir::Block::iterator(outer));
    mlir::BlockAndValueMapping mapper;
    // auto cloned = b.clone(*outer, mapper);
    b.clone(*outer, mapper);
    auto cloned = (--mlir::Block::iterator(outer));

    auto clonedFor = mlir::dyn_cast<mlir::AffineForOp>(cloned);
    assert(clonedFor);
    auto& ops_ = clonedFor.getBody()->getOperations();
  
    int count = 0;
    auto iter = --(--(ops_.end()));
    int number = ops_.size();
    for (int i = 0; i < number - position - 1; i++) {
      ++count;
      // it seems that iter->erase() will cause segment false.
      (iter--)->erase();
    }
  }
  if (existOpAfterLoop) {
    mlir::OpBuilder b(outer->getBlock(), ++mlir::Block::iterator(outer));
    mlir::BlockAndValueMapping mapper;
    auto cloned = b.clone(*outer, mapper);
    auto& ops_ = mlir::dyn_cast<mlir::AffineForOp>(cloned).getBody()->getOperations();
    auto iter = ops_.end();
    int number = ops_.size();
    for (int i = 0; i < number - position; i++) --iter;
    for(int i = 0; i <= position; i++) {
      (iter--)->erase();
    }
  }
  // clear current outer loop
  if (existOpBeforeLoop || existOpAfterLoop) {
    auto iter = --(ops.end());
    int number = ops.size();
    for (int i = 0; i < number; i++) {
      if (i == number - 1 - position || i == 0) {
        --iter;
      } else {
        (iter--)->erase();
      }
    }

  }
  // int count = 0;
  // for (auto& op : ops) {
  //   if (count != position && count !=  opNumber-1) {
  //     op.erase();
  //   }
  //   count += 1;
  // }


  /// step1: move the body of inner to outer
  // erase the yield op
  inner.getBody()->back().erase();
  // this block contain the inner Op
  inner->getBlock()->getOperations().splice( // this block is belong to outer
    mlir::Block::iterator(inner),
    inner.getBody()->getOperations());

  /// step2: move inner before outer.
  inner->moveBefore(outer);

  /// step3: make the outer as the body of inner
  inner.getBody()->getOperations().splice(inner.getBody()->end(),
                  outer->getBlock()->getOperations(), mlir::Block::iterator(outer));//only the outer.

  mlir::OpBuilder builder(inner.getContext());
  builder.setInsertionPointToEnd(inner.getBody());
  builder.create<mlir::AffineYieldOp>(builder.getUnknownLoc());
}

// Based on bubble sort.
// all loop in loops_ must be nested, but it can exists additional statement.
// Like this:
/*
for1, for2, for3 -----> for3, for1, for2
===origin
for1
	for2
		st1
		for3
		st2
===swap1
for1
	for2
		st1
	for3
		for2
	for2
		st2
===swap2
for1
	for2
		st1
for3
	for1
		for2
for1
	for2
		st2
*/
void Rewriter::reorder(const std::vector<mlir::AffineForOp>& loops) {

  // auto loops = loops_;
  // bufferizeLoopCarryVar(loops);
  // bufferizeLoopCarryVar(loops);
  // give every loop a prioriry
  std::map<mlir::AffineForOp, int, CompareLoop> loopPriority;
  int priority = loops.size();
  for (auto loop : loops) {
    loopPriority[loop] = priority--;
  }

  auto findFirstTargetLoop = [&](mlir::AffineForOp root) {
    if (loopPriority.count(root) != 0) return root;
    mlir::AffineForOp result;
    bool found = false;
    root.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineForOp forOp) {
      if ((!found) && loopPriority.count(forOp) != 0) {
        result = forOp;
        found = true;
      }
    });
    assert(found);
    return result;
  };

  auto containTargetLoop = [&](mlir::AffineForOp root) {

    auto& ops = root.getBody()->getOperations();
    mlir::AffineForOp sonLoop;
    bool result = false;

    for (auto& op : ops) {
      if (auto sonOp = mlir::dyn_cast<mlir::AffineForOp>(op)) {
        if (loopPriority.count(sonOp)) {
          result = true;
          sonLoop = sonOp;
          break;
        }
      }
    }
    return result ? sonLoop : nullptr;
  };

  bool swapped;

  mlir::AffineForOp rootForOp = findRootLoop(loops[0]);

  auto parentLoop_ = findFirstTargetLoop(rootForOp);

  // bubble sort.
  do {
    swapped = false;
    mlir::AffineForOp parentLoop = parentLoop_;
    while (auto sonLoop = containTargetLoop(parentLoop)) {
      if (loopPriority[parentLoop] < loopPriority[sonLoop]) {
        swap(parentLoop, sonLoop);
        swapped = true;
      } else {
        parentLoop = sonLoop;
      }
    }
  } while (swapped);
}

// op in forOps must be perfect nested loops.
mlir::AffineParallelOp Rewriter::parallel(const std::vector<mlir::AffineForOp>& forOps) {
  // X, Y, Z
  assert(forOps.size() <= 3);
  llvm::SmallVector<mlir::AffineMap> lbMaps;
  llvm::SmallVector<mlir::AffineMap> upMaps;
  llvm::SmallVector<mlir::Value> lbOperands;
  llvm::SmallVector<mlir::Value> upOperands;
  llvm::SmallVector<int64_t> steps;

  for (auto forOp : forOps) {
    lbMaps.push_back(forOp.getLowerBoundMap());
    upMaps.push_back(forOp.getUpperBoundMap());
    lbOperands.append(forOp.getLowerBoundOperands().begin(), forOp.getLowerBoundOperands().end());
    upOperands.append(forOp.getUpperBoundOperands().begin(), forOp.getUpperBoundOperands().end());
    steps.push_back(forOp.getStep());
  }

  mlir::OpBuilder builder(forOps[0]);

  mlir::AffineParallelOp parallelOp = builder.create<mlir::AffineParallelOp>(
    builder.getUnknownLoc(), mlir::TypeRange(), llvm::ArrayRef<mlir::arith::AtomicRMWKind>(),
    llvm::ArrayRef<mlir::AffineMap>(lbMaps), lbOperands,
    llvm::ArrayRef<mlir::AffineMap>(upMaps), upOperands,
    llvm::ArrayRef<int64_t>(steps));
  
  // erase the yield op of innermost loop
  auto innermost = forOps.back();
  innermost.getBody()->back().erase();
  // move the body of innermost loop to the begin of move
  parallelOp.getBody()->getOperations().splice(parallelOp.getBody()->begin(),
    innermost.getBody()->getOperations());

  auto newIvs = parallelOp.getIVs();
  int count = newIvs.size() - 1;

  for (auto iter = forOps.rbegin(); iter != forOps.rend(); ++iter) {
    auto forOp = *iter;
    forOp.getInductionVar().replaceAllUsesWith(newIvs[count--]);
    forOp.erase();
  }
  // make the lowerbound to 0 and step to 1
  mlir::normalizeAffineParallel(parallelOp);
  return parallelOp;
}

// dst is register.
mlir::AffineForOp Rewriter::read(mlir::Value src, mlir::Value dst, mlir::AffineMap map, 
                                   llvm::SmallVector<mlir::Value> operands, int64_t width,
                                   mlir::AffineForOp compute_at, Position pos) {
  auto dim0 = mlir::getAffineDimExpr(0, compute_at.getContext());
  auto dstMap = mlir::AffineMap::get(/*dimCount*/1, 0, llvm::ArrayRef<mlir::AffineExpr>(dim0 * width), 
                                     compute_at.getContext());
  auto builder = getBuilder(compute_at, pos);
  auto dstType = dst.getType().dyn_cast<mlir::MemRefType>();
  // registers is always 1 dim.
  auto loadTimes = dstType.getShape()[0] / width;
  auto loadBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv,
                      mlir::ValueRange iterArgs) {
    mlir::OpBuilder::InsertionGuard nestedGuard(builder);
    // loop iterator is the last operand.
    operands.push_back(iv);
    auto vectorType = mlir::VectorType::get(width, dstType.getElementType());
    auto ld = builder.create<mlir::AffineVectorLoadOp>(builder.getUnknownLoc(), vectorType, src, map, operands);
    auto st = builder.create<mlir::AffineVectorStoreOp>(builder.getUnknownLoc(), ld.getResult(), dst, dstMap, mlir::ValueRange({iv}));
    builder.create<mlir::AffineYieldOp>(builder.getUnknownLoc());
  };
  auto load = builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), 
     0, loadTimes, 1, /*iterArgs=lvm::None*/ mlir::ValueRange({}), loadBody);
  return load;
}

mlir::AffineForOp Rewriter::read(mlir::OpBuilder& builder, mlir::Value src, mlir::Value dst, 
    mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands, int64_t width) {
  auto dim0 = builder.getAffineDimExpr(0);
  auto dstMap = mlir::AffineMap::get(/*dimCount*/1, 0, llvm::ArrayRef<mlir::AffineExpr>(dim0 * width), 
                                     builder.getContext());
  auto dstType = dst.getType().dyn_cast<mlir::MemRefType>();
  // registers is always 1 dim.
  auto loadTimes = dstType.getShape()[0] / width;
  auto loadBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv,
                      mlir::ValueRange iterArgs) {
    mlir::OpBuilder::InsertionGuard nestedGuard(builder);
    // loop iterator is the last operand.
    operands.push_back(iv);
    auto vectorType = mlir::VectorType::get(width, dstType.getElementType());
    auto ld = builder.create<mlir::AffineVectorLoadOp>(builder.getUnknownLoc(), vectorType, src, map, operands);
    auto st = builder.create<mlir::AffineVectorStoreOp>(builder.getUnknownLoc(), ld.getResult(), dst, dstMap, mlir::ValueRange({iv}));
    builder.create<mlir::AffineYieldOp>(builder.getUnknownLoc());
  };
  auto load = builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), 
     0, loadTimes, 1, /*iterArgs=lvm::None*/ mlir::ValueRange({}), loadBody);
  return load;
}

// src is register
mlir::AffineForOp Rewriter::write(mlir::Value src, mlir::Value dst, mlir::AffineMap map, 
                                   llvm::SmallVector<mlir::Value> operands, int64_t width,
                                   mlir::AffineForOp compute_at, Position pos) {
  auto dimsNum = map.getNumDims();
  auto dim0 = mlir::getAffineDimExpr(0, compute_at.getContext());
  auto dim1 = mlir::getAffineDimExpr(1, compute_at.getContext());
  bool twoLoop = abs(dimsNum - operands.size()) == 2;
  auto srcMap = !twoLoop ? 
                mlir::AffineMap::get(/*dimCount*/1, 0, llvm::ArrayRef<mlir::AffineExpr>(dim0 * width), compute_at.getContext()) :
                mlir::AffineMap::get(/*dimCount*/2, 0, llvm::ArrayRef<mlir::AffineExpr>(dim0 * width + dim1), compute_at.getContext());
  auto builder = getBuilder(compute_at, pos);
  auto srcType = src.getType().dyn_cast<mlir::MemRefType>();
  // registers is always 1 dim.
  auto storeTimes = srcType.getShape()[0] / width;
  auto storeBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv,
                      mlir::ValueRange iterArgs) {
    mlir::OpBuilder::InsertionGuard nestedGuard(builder);
    // loop iterator is the last operand.
    operands.push_back(iv);
    if (twoLoop) {
      auto innerBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv_inner,
                        mlir::ValueRange iterArgs) {
        mlir::OpBuilder::InsertionGuard nestedGuard(builder);
        // loop iterator is the last operand.
        operands.push_back(iv_inner);
        auto vectorType = mlir::VectorType::get(1, srcType.getElementType());
        auto ld = builder.create<mlir::AffineVectorLoadOp>(builder.getUnknownLoc(), vectorType, src, srcMap, mlir::ValueRange({iv, iv_inner}));
        auto st = builder.create<mlir::AffineVectorStoreOp>(builder.getUnknownLoc(), ld.getResult(), dst, map, operands);
        builder.create<mlir::AffineYieldOp>(builder.getUnknownLoc());
      };
      auto storeInner = builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), 
          0, width, 1, /*iterArgs=lvm::None*/ mlir::ValueRange({}), innerBody);
      builder.create<mlir::AffineYieldOp>(builder.getUnknownLoc());
    } else { 
      auto vectorType = mlir::VectorType::get(width, srcType.getElementType());
      auto ld = builder.create<mlir::AffineVectorLoadOp>(builder.getUnknownLoc(), vectorType, src, srcMap, mlir::ValueRange({iv}));
      auto st = builder.create<mlir::AffineVectorStoreOp>(builder.getUnknownLoc(), ld.getResult(), dst, map, operands);
      builder.create<mlir::AffineYieldOp>(builder.getUnknownLoc());
    }
  };
  auto store = builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), 
     0, storeTimes, 1, /*iterArgs=lvm::None*/ mlir::ValueRange({}), storeBody);
  return store;

}

// src is register
mlir::AffineForOp Rewriter::write(mlir::OpBuilder& builder, mlir::Value src, mlir::Value dst, 
    mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands, int64_t width) {
  auto dimsNum = map.getNumDims();
  auto dim0 = builder.getAffineDimExpr(0);
  auto dim1 = builder.getAffineDimExpr(1);
  bool twoLoop = abs(dimsNum - operands.size()) == 2;
  auto srcMap = !twoLoop ? 
                mlir::AffineMap::get(/*dimCount*/1, 0, llvm::ArrayRef<mlir::AffineExpr>(dim0 * width), builder.getContext()) :
                mlir::AffineMap::get(/*dimCount*/2, 0, llvm::ArrayRef<mlir::AffineExpr>(dim0 * width + dim1), builder.getContext());
  auto srcType = src.getType().dyn_cast<mlir::MemRefType>();
  // registers is always 1 dim.
  auto storeTimes = srcType.getShape()[0] / width;
  auto storeBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv,
                      mlir::ValueRange iterArgs) {
    mlir::OpBuilder::InsertionGuard nestedGuard(builder);
    // loop iterator is the last operand.
    operands.push_back(iv);
    if (twoLoop) {
      auto innerBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv_inner,
                        mlir::ValueRange iterArgs) {
        mlir::OpBuilder::InsertionGuard nestedGuard(builder);
        // loop iterator is the last operand.
        operands.push_back(iv_inner);
        auto vectorType = mlir::VectorType::get(1, srcType.getElementType());
        auto ld = builder.create<mlir::AffineVectorLoadOp>(builder.getUnknownLoc(), vectorType, src, srcMap, mlir::ValueRange({iv, iv_inner}));
        auto st = builder.create<mlir::AffineVectorStoreOp>(builder.getUnknownLoc(), ld.getResult(), dst, map, operands);
        builder.create<mlir::AffineYieldOp>(builder.getUnknownLoc());
      };
      auto storeInner = builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), 
          0, width, 1, /*iterArgs=lvm::None*/ mlir::ValueRange({}), innerBody);
      builder.create<mlir::AffineYieldOp>(builder.getUnknownLoc());
    } else { 
      auto vectorType = mlir::VectorType::get(width, srcType.getElementType());
      auto ld = builder.create<mlir::AffineVectorLoadOp>(builder.getUnknownLoc(), vectorType, src, srcMap, mlir::ValueRange({iv}));
      auto st = builder.create<mlir::AffineVectorStoreOp>(builder.getUnknownLoc(), ld.getResult(), dst, map, operands);
      builder.create<mlir::AffineYieldOp>(builder.getUnknownLoc());
    }
  };
  auto store = builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), 
     0, storeTimes, 1, /*iterArgs=lvm::None*/ mlir::ValueRange({}), storeBody);
  return store;
}

mlir::gpu::BarrierOp Rewriter::barrier(mlir::AffineForOp compute_at, Position pos) {
  auto builder = getBuilder(compute_at, pos);
  return builder.create<mlir::gpu::BarrierOp>(builder.getUnknownLoc());
}

void Rewriter::cache_read(mlir::AffineForOp scope, mlir::Value src, mlir::Value cached, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands) {
  scope.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineLoadOp load) {
    if (load.getMemref() != src) return;
    mlir::OpBuilder builder(load);
    auto newLoad = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), cached, map, operands);
    load.getResult().replaceAllUsesWith(newLoad.getResult());
    load.erase();
  });
}

void Rewriter::cache_write(mlir::AffineForOp scope, mlir::Value src, mlir::Value cached, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands) {
  scope.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineStoreOp store) {
    if (store.getMemref() != src) return;
    mlir::OpBuilder builder(store);
    auto newStore = builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), store.getValue(), src, map, operands);
    store.erase();
  });
}

///TODO: two level vector.
std::vector<std::vector<mlir::AffineForOp>> Rewriter::get_write(mlir::AffineParallelOp parallelLevel, mlir::Value dst) {
  std::vector<std::vector<mlir::AffineForOp>> results;
  std::vector<mlir::AffineStoreOp> stores;
  parallelLevel.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineStoreOp store) {
    if (store.getMemref() != dst) return;
    stores.push_back(store);
  });
  for (auto store : stores) {
    std::vector<mlir::AffineForOp> result;
    mlir::AffineForOp parent;
    mlir::Operation* cur = store;
    while (parent = mlir::dyn_cast<mlir::AffineForOp>(cur->getParentOp())) {
      result.push_back(parent);
      cur = parent;
    }
    std::reverse(result.begin(), result.end());
    results.push_back(result);
  }
  return results;
}

mlir::AffineForOp Rewriter::vectorize(mlir::AffineForOp readOrWrite, int64_t width) {
  int64_t step = readOrWrite.getStep();
  int64_t ub = readOrWrite.getConstantUpperBound();
  int64_t lb = readOrWrite.getConstantLowerBound();
  assert(step = 1 && lb == 0 && ub % width == 0);
  readOrWrite.setStep(width);
  readOrWrite.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineLoadOp load) {
    mlir::OpBuilder builder(load);
    auto type = load.getMemRef().getType().dyn_cast<mlir::MemRefType>();
    auto vectorType = mlir::VectorType::get(width, type.getElementType());
    auto vectorLoad = builder.create<mlir::AffineVectorLoadOp>(builder.getUnknownLoc(), vectorType, load.getMemRef(), load.getAffineMap(), load.getMapOperands());
    load.getResult().replaceAllUsesWith(vectorLoad.getResult());
    load.erase();
  });
  readOrWrite.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineStoreOp store) {
    mlir::OpBuilder builder(store);
     auto type = store.getMemRef().getType().dyn_cast<mlir::MemRefType>();
    auto vectorType = mlir::VectorType::get(width, type.getElementType());
    auto vectorStore = builder.create<mlir::AffineVectorStoreOp>(builder.getUnknownLoc(), store.getValue(), store.getMemRef(), store.getAffineMap(), store.getMapOperands());
    store.erase();
  });
}

std::vector<std::vector<mlir::AffineForOp>> Rewriter::pipeline(std::vector<mlir::AffineForOp> readBodys, mlir::Value& buffer, mlir::AffineForOp compute_at) {

  // bool shared;
  // if (memorySpace == static_cast<int>(MemorySpace::shared)) {
  //   shared = true;
  //   assert(readBodys.size() == 2);
  // } else {
  //   shared = false;
  //   assert(readBodys.size() == 1);
  // }

  std::vector<std::vector<mlir::AffineForOp>> results;

  /* step1: double buffer.*/

  auto bufferType = buffer.getType().dyn_cast<mlir::MemRefType>();
  mlir::SmallVector<int64_t> shape;
  /// double size on top dim.
  shape.push_back(2);
  for (auto dim : bufferType.getShape()) {
    shape.push_back(dim);
  }
  auto newBufferType = mlir::MemRefType::get(
    shape, bufferType.getElementType(), {}, bufferType.getMemorySpaceAsInt());
  auto defineBufferOp = mlir::dyn_cast<mlir::memref::AllocOp>(buffer.getDefiningOp());
  mlir::OpBuilder builder(defineBufferOp);
  auto allocOp = builder.create<mlir::memref::AllocOp>(
    builder.getUnknownLoc(), newBufferType);
  auto doubleBuffer = allocOp.getResult();


  /* step2: prefetch before the loop.*/
  //1. replace every use of compute_at's inductionvar with compute_at'lb.
  auto replaceOperand = [&](mlir::AffineForOp body, mlir::Value src, mlir::Value dst) {
    body.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineVectorLoadOp load) {
      auto oldOperands = load.getMapOperands();
      mlir::SmallVector<mlir::Value> operands;
      bool needReplace = false;
      for (auto operand : oldOperands) {
        if (operand == src) {
          needReplace = true;
          operands.push_back(dst);
        } else {
          operands.push_back(operand);
        }
      }
      if (!needReplace) return;
      mlir::OpBuilder builder(load);
      auto newVectorLoadOp = builder.create<mlir::AffineVectorLoadOp>(builder.getUnknownLoc(), load.getVectorType(), load.getMemref(), load.getAffineMap(), operands);
      load.getResult().replaceAllUsesWith(newVectorLoadOp.getResult());
      load.erase();
    });
    body.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineVectorStoreOp store) {
      auto oldOperands = store.getMapOperands();
      mlir::SmallVector<mlir::Value> operands;
      bool needReplace = false;
      for (auto operand : oldOperands) {
        if (operand == src) {
          needReplace = true;
          operands.push_back(dst);
        } else {
          operands.push_back(operand);
        }
      }
      if (!needReplace) return;
      mlir::OpBuilder builder(store);
      auto newVectorStoreOp = builder.create<mlir::AffineVectorStoreOp>(builder.getUnknownLoc(), store.getValue(), store.getMemref(), store.getAffineMap(), operands);
      store.erase();
    });
  };
  //2. replace every reference to buffer with doubleBuffer, and select doubleBuffer[0];
  auto replaceBufferRef = [&](mlir::AffineForOp body, mlir::Value bufferSrc, mlir::Value bufferDst) {
    body.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineVectorLoadOp load) {
      auto oldMemref = load.getMemref();
      if (oldMemref != bufferSrc) return;

      auto oldAffineMap = load.getAffineMap();
      auto oldExprs = oldAffineMap.getResults();
      mlir::SmallVector<mlir::AffineExpr> exprs;
      exprs.push_back(mlir::getAffineConstantExpr(0, body->getContext()));
      for (auto expr : oldExprs) exprs.push_back(expr);
      auto map = mlir::AffineMap::get(/*dimCount*/oldAffineMap.getNumDims(), 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), body->getContext());

      mlir::OpBuilder builder(load);
      auto newVectorLoadOp = builder.create<mlir::AffineVectorLoadOp>(builder.getUnknownLoc(), load.getVectorType(), bufferDst, map, load.getMapOperands());
      load.getResult().replaceAllUsesWith(newVectorLoadOp.getResult());
      load.erase();
    });
    body.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineVectorStoreOp store) {
      auto oldMemref = store.getMemref();
      if (oldMemref != bufferSrc) return;

      auto oldAffineMap = store.getAffineMap();
      auto oldExprs = oldAffineMap.getResults();
      mlir::SmallVector<mlir::AffineExpr> exprs;
      exprs.push_back(mlir::getAffineConstantExpr(0, body->getContext()));
      for (auto expr : oldExprs) exprs.push_back(expr);
      auto map = mlir::AffineMap::get(/*dimCount*/oldAffineMap.getNumDims(), 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), body->getContext());

      mlir::OpBuilder builder(store);
      auto newVectorStoreOp = builder.create<mlir::AffineVectorStoreOp>(builder.getUnknownLoc(), store.getValue(), bufferDst, map, store.getMapOperands());
      store.erase();
    });
  };
  std::vector<mlir::AffineForOp> result;
  builder.setInsertionPoint(compute_at);
  auto lbOp = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), compute_at.getConstantLowerBound());
  auto rootLoop = findRootLoop(compute_at);
  lbOp->moveBefore(&(rootLoop->getBlock()->getOperations().front()));
  for (auto readBody : readBodys) {
    mlir::BlockAndValueMapping mapper;
    auto newBody = builder.clone(*readBody, mapper);
    auto loopBody = mlir::dyn_cast<mlir::AffineForOp>(newBody);
    replaceOperand(loopBody, compute_at.getInductionVar(), lbOp.getResult());
    replaceBufferRef(loopBody, buffer, doubleBuffer);
    result.push_back(loopBody);
  }
  results.push_back(result);
  results.push_back(readBodys);


  /* step3: prefetch in the main loop*/
  //1. create the affine.if to check if we can prefetch
  auto dim0 = builder.getAffineDimExpr(0);
  auto dim1 = builder.getAffineDimExpr(1);

  int64_t step = compute_at.getStep();
  int64_t ub = compute_at.getConstantUpperBound();
  int64_t lb = compute_at.getConstantLowerBound();

  /*
  /// Array of affine constraints: a constraint is either an equality
  /// (affine_expr == 0) or an inequality (affine_expr >= 0).
  ArrayRef<AffineExpr> constraints;

  // Bits to check whether a constraint is an equality or an inequality.
  ArrayRef<bool> eqFlags;
  */
  llvm::SmallVector<mlir::AffineExpr> exprs;
  llvm::SmallVector<bool> eqFlags;
  // iv + 2 * step <= ub
  //-> ub - 2 * step - iv >= 0
  exprs.push_back(ub - 2 * step - dim0);
  eqFlags.push_back(false);
  auto cst = mlir::IntegerSet::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), llvm::ArrayRef<bool>(eqFlags));

  builder.setInsertionPointToStart(compute_at.getBody());
  auto ifOp = builder.create<mlir::AffineIfOp>(builder.getUnknownLoc(), cst, mlir::ValueRange{compute_at.getInductionVar()}, 
                                               /*withElseRegion=*/false);
  
  builder.setInsertionPointToStart(ifOp.getThenBlock());

  auto reverseReadBodys = readBodys;
  std::reverse(reverseReadBodys.begin(), reverseReadBodys.end());

  for (auto readBody : reverseReadBodys) {
    ifOp.getBody()->getOperations().splice(ifOp.getBody()->begin(),
                    readBody->getBlock()->getOperations(), mlir::Block::iterator(readBody));//only the readBody.
  }
  // 2. replace 
  auto replaceAffineExprInLoop = [&](mlir::AffineForOp body, mlir::Value src, mlir::AffineExpr dstExpr, int dimCount) {
    body.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineVectorLoadOp load) {
      auto operands = load.getMapOperands();
      bool needReplace = false;
      int targetDim = -1;
      for (auto operand : operands) {
        if (!needReplace) targetDim += 1;
        if (operand == src) {
          needReplace = true;
          break;
        }
      }
      if (!needReplace) return;
      auto shiftedDstExpr = shiftAffineExprDim(body->getContext(), dstExpr, targetDim);
      llvm::SmallVector<mlir::AffineExpr> exprs;
      auto oldExprs = load.getAffineMap().getResults();
      for (auto oldExpr : oldExprs) {
        auto expr = getModifiedExpr(body->getContext(), oldExpr, shiftedDstExpr, targetDim, dimCount);
        exprs.push_back(expr);
      }
      auto map = mlir::AffineMap::get(/*dimCount*/load.getAffineMap().getNumDims() + dimCount - 1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), body->getContext());
      mlir::OpBuilder builder(load);
      auto newVectorLoadOp = builder.create<mlir::AffineVectorLoadOp>(builder.getUnknownLoc(), load.getVectorType(), load.getMemref(), map, operands);
      load.getResult().replaceAllUsesWith(newVectorLoadOp.getResult());
      load.erase();
    });
    body.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineVectorStoreOp store) {
      auto operands = store.getMapOperands();
      bool needReplace = false;
      int targetDim = -1;
      for (auto operand : operands) {
        if (!needReplace) targetDim += 1;
        if (operand == src) {
          needReplace = true;
          break;
        }
      }
      if (!needReplace) return;
      auto shiftedDstExpr = shiftAffineExprDim(body->getContext(), dstExpr, targetDim);
      llvm::SmallVector<mlir::AffineExpr> exprs;
      auto oldExprs = store.getAffineMap().getResults();
      for (auto oldExpr : oldExprs) {
        auto expr = getModifiedExpr(body->getContext(), oldExpr, shiftedDstExpr, targetDim, dimCount);
        exprs.push_back(expr);
      }
      auto map = mlir::AffineMap::get(/*dimCount*/store.getAffineMap().getNumDims() + dimCount - 1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), body->getContext());
      mlir::OpBuilder builder(store);
      auto newVectorStoreOp = builder.create<mlir::AffineVectorStoreOp>(builder.getUnknownLoc(), store.getValue(), store.getMemref(), map, operands);
      store.erase();
    });
  };
  // 3.replace every reference to buffer with doubleBuffer, and select doubleBuffer[0];
  auto replaceBufferRefInLoop = [&](mlir::AffineForOp body, mlir::Value bufferSrc, mlir::Value bufferDst, mlir::AffineForOp compute_at) {
    body.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineVectorLoadOp load) {
      auto oldMemref = load.getMemref();
      if (oldMemref != bufferSrc) return;

      int targetDim = -1;
      int additionDim = 0;
      bool existInductionVar = false;
      for (auto operand : load.getMapOperands()) {
        targetDim += 1;
        if (operand == compute_at.getInductionVar()) {
          existInductionVar = true;
          break;
        }
      }
      // reference to double buffer must depend on the iteration var.
      llvm::SmallVector<mlir::Value> operands;
      for (auto operand : load.getMapOperands()) operands.push_back(operand);
      if (!existInductionVar) {
        operands.push_back(compute_at.getInductionVar());
        targetDim += 1;
        additionDim += 1;
      }
      auto dim = mlir::getAffineDimExpr(targetDim, body->getContext());
      mlir::SmallVector<mlir::AffineExpr> exprs;
      exprs.push_back((dim.floorDiv(compute_at.getStep()) + 1) % 2);
      auto oldAffineMap = load.getAffineMap();
      auto oldExprs = oldAffineMap.getResults();
      for (auto expr : oldExprs) exprs.push_back(expr);
      auto map = mlir::AffineMap::get(/*dimCount*/oldAffineMap.getNumDims() + additionDim, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), body->getContext());

      mlir::OpBuilder builder(load);
      auto newVectorLoadOp = builder.create<mlir::AffineVectorLoadOp>(builder.getUnknownLoc(), load.getVectorType(), bufferDst, map, operands);
      load.getResult().replaceAllUsesWith(newVectorLoadOp.getResult());
      load.erase();
    });
    body.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineVectorStoreOp store) {
      auto oldMemref = store.getMemref();
      if (oldMemref != bufferSrc) return;

      int targetDim = -1;
      int additionDim = 0;
      bool existInductionVar = false;
      for (auto operand : store.getMapOperands()) {
        targetDim += 1;
        if (operand == compute_at.getInductionVar()) {
          existInductionVar = true;
          break;
        }
      }
      // reference to double buffer must depend on the iteration var.
      llvm::SmallVector<mlir::Value> operands;
      for (auto operand : store.getMapOperands()) operands.push_back(operand);
      if (!existInductionVar) {
        operands.push_back(compute_at.getInductionVar());
        targetDim += 1;
        additionDim += 1;
      }
      auto dim = mlir::getAffineDimExpr(targetDim, body->getContext());
      mlir::SmallVector<mlir::AffineExpr> exprs;
      exprs.push_back((dim.floorDiv(compute_at.getStep()) + 1) % 2);
      auto oldAffineMap = store.getAffineMap();
      auto oldExprs = oldAffineMap.getResults();
      for (auto expr : oldExprs) exprs.push_back(expr);
      auto map = mlir::AffineMap::get(/*dimCount*/oldAffineMap.getNumDims() + additionDim, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), body->getContext());

      mlir::OpBuilder builder(store);
      auto newVectorStoreOp = builder.create<mlir::AffineVectorStoreOp>(builder.getUnknownLoc(), store.getValue(), bufferDst, map, operands);
      store.erase();
    });
  };
  for (auto readBody : readBodys) {
    auto dim0 = builder.getAffineDimExpr(0);
    replaceAffineExprInLoop(readBody, compute_at.getInductionVar(), dim0 + compute_at.getStep(), 1);
    replaceBufferRefInLoop(readBody, buffer, doubleBuffer, compute_at);
  }
  //4. replace load
  auto users = buffer.getUsers();
  for (auto user : users) {
    // must be load Op
    if (auto load = mlir::dyn_cast<mlir::AffineVectorLoadOp>(user)) {
      auto oldMemref = load.getMemref();
      if (oldMemref != buffer) assert(false);

      int targetDim = -1;
      int additionDim = 0;
      bool existInductionVar = false;
      for (auto operand : load.getMapOperands()) {
        targetDim += 1;
        if (operand == compute_at.getInductionVar()) {
          existInductionVar = true;
          break;
        }
      }
      // reference to double buffer must depend on the iteration var.
      llvm::SmallVector<mlir::Value> operands;
      for (auto operand : load.getMapOperands()) operands.push_back(operand);
      if (!existInductionVar) {
        operands.push_back(compute_at.getInductionVar());
        targetDim += 1;
        additionDim += 1;
      }
      auto dim = mlir::getAffineDimExpr(targetDim, load->getContext());
      mlir::SmallVector<mlir::AffineExpr> exprs;
      exprs.push_back(dim.floorDiv(compute_at.getStep()) % 2);
      auto oldAffineMap = load.getAffineMap();
      auto oldExprs = oldAffineMap.getResults();
      for (auto expr : oldExprs) exprs.push_back(expr);
      auto map = mlir::AffineMap::get(/*dimCount*/oldAffineMap.getNumDims() + additionDim, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), load->getContext());

      mlir::OpBuilder builder(load);
      auto newVectorLoadOp = builder.create<mlir::AffineVectorLoadOp>(builder.getUnknownLoc(), load.getVectorType(), doubleBuffer, map, operands);
      load.getResult().replaceAllUsesWith(newVectorLoadOp.getResult());
      load.erase();
    } else if (auto load = mlir::dyn_cast<mlir::AffineLoadOp>(user)) {
      auto oldMemref = load.getMemref();
      if (oldMemref != buffer) assert(false);

      int targetDim = -1;
      int additionDim = 0;
      bool existInductionVar = false;
      for (auto operand : load.getMapOperands()) {
        targetDim += 1;
        if (operand == compute_at.getInductionVar()) {
          existInductionVar = true;
          break;
        }
      }
      // reference to double buffer must depend on the iteration var.
      llvm::SmallVector<mlir::Value> operands;
      for (auto operand : load.getMapOperands()) operands.push_back(operand);
      if (!existInductionVar) {
        operands.push_back(compute_at.getInductionVar());
        targetDim += 1;
        additionDim += 1;
      }
      auto dim = mlir::getAffineDimExpr(targetDim, load->getContext());
      mlir::SmallVector<mlir::AffineExpr> exprs;
      exprs.push_back(dim.floorDiv(compute_at.getStep()) % 2);
      auto oldAffineMap = load.getAffineMap();
      auto oldExprs = oldAffineMap.getResults();
      for (auto expr : oldExprs) exprs.push_back(expr);
      auto map = mlir::AffineMap::get(/*dimCount*/oldAffineMap.getNumDims() + additionDim, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), load->getContext());

      mlir::OpBuilder builder(load);
      auto newVectorLoadOp = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), doubleBuffer, map, operands);
      load.getResult().replaceAllUsesWith(newVectorLoadOp.getResult());
      load.erase();
    } else {
      assert(false);
    }
  }

  /* step4: clear work*/
  defineBufferOp.erase();
  buffer = doubleBuffer;


  return results;
}

void Rewriter::detach_last_loop(mlir::AffineForOp forOp) {
  auto step = forOp.getStep();
  auto ub = forOp.getConstantUpperBound();
  forOp.setConstantUpperBound(ub - step);

  auto builder = getBuilder(forOp, Position::after);
  auto replaceInducetionVar = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), ub - step);
  auto rootLoop = findRootLoop(forOp);
  replaceInducetionVar->moveBefore(&(rootLoop->getBlock()->getOperations().front()));
  mlir::BlockAndValueMapping mapper;
  auto newBody = builder.clone(*forOp, mapper);
  auto loopBody = mlir::dyn_cast<mlir::AffineForOp>(newBody);
  loopBody.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
    auto oldOperands = op->getOperands();
    llvm::SmallVector<mlir::Value> operands;
    for (auto operand : oldOperands) {
      if (operand == loopBody.getInductionVar()) {
        operands.push_back(replaceInducetionVar.getResult());
      } else {
        operands.push_back(operand);
      }
    }
    op->setOperands(operands);
  });

  loopBody.getBody()->getOperations().back().erase();
  loopBody->getBlock()->getOperations().splice( 
    mlir::Block::iterator(loopBody),
    loopBody.getBody()->getOperations());
  loopBody.erase();

}

void Rewriter::schedule(mlir::Operation* srcOp, mlir::Operation* dstOp, Position pos) {
  mlir::OpBuilder builder(dstOp->getContext());
  switch (pos) {
    case Position::after: {
      builder.setInsertionPointAfter(dstOp);
      srcOp->moveAfter(dstOp);
      break;
    }
    case Position::before: {
      builder.setInsertionPoint(dstOp);
      srcOp->moveBefore(dstOp);
      break;
    }
    case Position::end: {
      if (auto forOp = mlir::dyn_cast<mlir::AffineForOp>(dstOp)) {
        srcOp->moveBefore(&(forOp.getBody()->getOperations().back()));
      } else {
        assert(false);
      }
      break;
    }
    default:
      assert(false);
  }

}

void replaceOperands(mlir::Operation* op, mlir::Value src, mlir::Value dst) {
  auto oldOperands = op->getOperands();
  llvm::SmallVector<mlir::Value> operands;
  for (auto operand : oldOperands) {
    if (operand == src) {
      operands.push_back(dst);
    } else {
      operands.push_back(operand);
    }
  }
  op->setOperands(operands);

  if (op->getRegions().size() != 0) {
    auto& blocks = op->getRegions().front().getBlocks();
    for (auto& block : blocks) {
      auto& ops = block.getOperations();
      for (auto& op : ops) {
        replaceOperands(&op, src, dst);
      }
    }
  }

}

void Rewriter::extract_loop(mlir::Operation* srcOp, mlir::AffineForOp forOp, int64_t iteration) {
  mlir::OpBuilder builder(forOp->getContext());
  builder.setInsertionPoint(forOp);
  mlir::BlockAndValueMapping mapper;
  auto clonedOp = builder.clone(*srcOp, mapper);

  int64_t step = forOp.getStep();
  int64_t ub = forOp.getConstantUpperBound();
  int64_t lb = forOp.getConstantLowerBound();

  auto index = lb + iteration * step;
  auto replaceVar = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), index);

  auto rootLoop = findRootLoop(forOp);
  replaceVar->moveBefore(&(rootLoop->getBlock()->getOperations().front()));

  replaceOperands(clonedOp, forOp.getInductionVar(), replaceVar.getResult());

}

std::pair<bool, int64_t> getMaxValue(mlir::Value value) {
  mlir::Operation* op;
  if (auto blockArgument = value.dyn_cast<mlir::BlockArgument>()) {
    op = blockArgument.getOwner()->getParentOp();
  } else {
    op = value.getDefiningOp();
  }
  auto constOp = mlir::dyn_cast<mlir::arith::ConstantOp>(op);
  std::pair<bool, int64_t> result;
  if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(op)) {
    result.first = true;
    result.second = constOp.value();
    // result.second = constOp.getValue().cast<mlir::IntegerAttr>().getInt();
  } else if (auto forOp = mlir::dyn_cast<mlir::AffineForOp>(op)) {
    if (!forOp.hasConstantBounds()) {
      result.first = false;
    } else {
      result.first = true;
      result.second = forOp.getConstantUpperBound() - 1;
    }
  } else {
    llvm::errs() << "Append new op type here.";
    assert(false);
  }
  return result;
}

std::pair<bool, int64_t> getMinValue(mlir::Value value) {
  mlir::Operation* op;
  if (auto blockArgument = value.dyn_cast<mlir::BlockArgument>()) {
    op = blockArgument.getOwner()->getParentOp();
  } else {
    op = value.getDefiningOp();
  }
  std::pair<bool, int64_t> result;
  if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(op)) {
    result.first = true;
    result.second = constOp.value();
  } else if (auto forOp = mlir::dyn_cast<mlir::AffineForOp>(op)) {
    if (!forOp.hasConstantBounds()) {
      result.first = false;
    } else {
      result.first = true;
      result.second = forOp.getConstantLowerBound();
    }
  } else {
    llvm::errs() << "Append new op type here.";
    assert(false);
  }
  return result;
}

int64_t eval(mlir::AffineExpr expr, std::vector<int64_t> values) {
  if (auto dimExpr = expr.dyn_cast<mlir::AffineDimExpr>()) {
    return values[dimExpr.getPosition()];
  }
  if (auto constExpr = expr.dyn_cast<mlir::AffineConstantExpr>()) {
    return constExpr.getValue();
  }
  auto binaryExpr = expr.dyn_cast<mlir::AffineBinaryOpExpr>();
  assert(binaryExpr);
  auto lhs = eval(binaryExpr.getLHS(), values);
  auto rhs = eval(binaryExpr.getRHS(), values);
  switch (binaryExpr.getKind()) {
    case mlir::AffineExprKind::Add: return lhs + rhs;
    case mlir::AffineExprKind::CeilDiv: return (lhs + rhs - 1) / rhs;
    case mlir::AffineExprKind::FloorDiv: return lhs / rhs;
    case mlir::AffineExprKind::Mod: return lhs % rhs;
    case mlir::AffineExprKind::Mul: return lhs * rhs;
    default: assert(false);
  }
}

struct TakeOffTrueIf : 
  public mlir::PassWrapper<TakeOffTrueIf, mlir::OperationPass<mlir::ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TakeOffTrueIf)
   TakeOffTrueIf() = default;
   void runOnOperation() override {
     auto module = getOperation();
     module.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineIfOp ifOp) {
      bool result = true;
      auto iset = ifOp.getIntegerSet();
      auto operands = ifOp->getOperands();

      int constraintNum = iset.getNumConstraints();
      std::vector<int64_t> maxValues;
      std::vector<int64_t> minValues;

      for (auto operand : operands) {
        auto maxValue = getMaxValue(operand);
        if (maxValue.first) {
          maxValues.push_back(maxValue.second);
        } else {
          //can't deduction
          return;
        }
        auto minValue = getMinValue(operand);
        if (minValue.first) {
          minValues.push_back(minValue.second);
        } else {
          //can't deduction
          return;
        }
      }
      for (int i = 0; i < constraintNum; i++) {
        auto expr = iset.getConstraint(i);
        auto isEq = iset.isEq(i);
        if (isEq) {
          if (eval(expr, maxValues) != 0 | eval(expr, minValues) != 0) {
            result = false;
            break;
          }
        } else {
          if (eval(expr, maxValues) < 0 | eval(expr, minValues) < 0) {
            result = false;
            break;
          }
        }
      }
      if (result) {
        ifOp.getBody()->getOperations().back().erase();
        ifOp->getBlock()->getOperations().splice(
          mlir::Block::iterator(ifOp),
          ifOp.getBody()->getOperations());
        ifOp.erase();
      }
     });
   }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> 
TakeOffTrueIfPass() {
  return std::make_unique<TakeOffTrueIf>();
}

void Rewriter::take_off_true_if(mlir::ModuleOp module) {
  mlir::PassManager pm(module.getContext());
  pm.addPass(TakeOffTrueIfPass());
  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Take off the true if failed.";
  }
  return;
}

struct DeleteFalseIf : 
  public mlir::PassWrapper<DeleteFalseIf, mlir::OperationPass<mlir::ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeleteFalseIf)
   DeleteFalseIf() = default;
   void runOnOperation() override {
     auto module = getOperation();
     module.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineIfOp ifOp) {
      auto iset = ifOp.getIntegerSet();
      auto operands = ifOp->getOperands();

      int constraintNum = iset.getNumConstraints();
      std::vector<int64_t> maxValues;
      std::vector<int64_t> minValues;

      for (auto operand : operands) {
        auto maxValue = getMaxValue(operand);
        if (maxValue.first) {
          maxValues.push_back(maxValue.second);
        } else {
          //can't deduction
          return;
        }
        auto minValue = getMinValue(operand);
        if (minValue.first) {
          minValues.push_back(minValue.second);
        } else {
          //can't deduction
          return;
        }
      }
      int64_t count = 0;
      for (int i = 0; i < constraintNum; i++) {
        auto expr = iset.getConstraint(i);
        auto isEq = iset.isEq(i);
        ///TODO:need to verify all the case of all inputs.
        if (isEq) {
          if (eval(expr, maxValues) != 0 && eval(expr, minValues) != 0) {
            count += 1;
          }
        } else {
          if (eval(expr, maxValues) < 0 && eval(expr, minValues) < 0) {
            count += 1;
          }
        }
      }
      if (count == constraintNum) {
        // delete the entile body of if operaiton.
        ifOp.erase();
      }
     });
   }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> 
DeleteFalseIfPass() {
  return std::make_unique<DeleteFalseIf>();
}

void Rewriter::delete_false_if(mlir::ModuleOp module) {
  mlir::PassManager pm(module.getContext());
  pm.addPass(DeleteFalseIfPass());
  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Delete false if failed.";
  }
  return;
}

struct UnrollAffineFor : 
  public mlir::PassWrapper<UnrollAffineFor, mlir::OperationPass<mlir::ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnrollAffineFor)
   UnrollAffineFor(mlir::function_ref<bool(mlir::AffineForOp)> unrollCheckFn_ = nullptr) : unrollCheckFn(unrollCheckFn_) {};
   void runOnOperation() override {

     auto module = getOperation();
     module.walk<mlir::WalkOrder::PostOrder>([&](mlir::AffineForOp forOp) {
      if (!unrollCheckFn(forOp)) return;

      auto rootLoop = findRootLoop(forOp);
      auto& allOps = rootLoop->getBlock()->getOperations();

      auto findConstValue = [&](int64_t value)->mlir::Value {
        auto curIter = allOps.begin();
        while (true) {
          auto constOp = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(*curIter);
          if (!constOp) break;
          if (value == constOp.value()) {
            return constOp.getResult();
          }
          curIter++;
        }
        return nullptr;
      };

      mlir::OpBuilder builder(forOp);

      for (auto index = forOp.getConstantLowerBound(); index < forOp.getConstantUpperBound(); index += forOp.getStep()) {
        auto iterVarReplace = findConstValue(index);
        if (!iterVarReplace) {
          auto constOp = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), index);
          constOp->moveBefore(&(rootLoop->getBlock()->getOperations().front()));
          iterVarReplace = constOp.getResult();
        }
        mlir::BlockAndValueMapping mapper;
        auto cloned = builder.clone(*forOp, mapper);
        auto clonedForOp = mlir::dyn_cast<mlir::AffineForOp>(cloned);
        clonedForOp.getBody()->getOperations().back().erase();
        clonedForOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
          auto oldOperands = op->getOperands();
          llvm::SmallVector<mlir::Value> operands;
          for (auto operand : oldOperands) {
            if (operand == clonedForOp.getInductionVar()) {
              operands.push_back(iterVarReplace);
            } else {
              operands.push_back(operand);
            }
          }
          op->setOperands(operands);
        });
        clonedForOp->getBlock()->getOperations().splice(
          mlir::Block::iterator(clonedForOp),
          clonedForOp.getBody()->getOperations());
        clonedForOp.erase();
      }
      forOp.erase();
     });
   }
  mlir::function_ref<bool(mlir::AffineForOp)> unrollCheckFn;
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> 
UnrollAffineForPass(mlir::function_ref<bool(mlir::AffineForOp)> unrollCheckFn = nullptr) {
  return std::make_unique<UnrollAffineFor>(unrollCheckFn);
}

void Rewriter::unroll(mlir::ModuleOp module, mlir::function_ref<bool(mlir::AffineForOp)> unrollCheckFn) {

  mlir::PassManager pm(module.getContext());
  pm.addPass(UnrollAffineForPass(unrollCheckFn));
  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Unroll affine for failed.";
  }
  return;
}

struct UnrollAttribute : 
  public mlir::PassWrapper<UnrollAttribute, mlir::OperationPass<mlir::ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnrollAttribute)
   UnrollAttribute(mlir::function_ref<bool(mlir::AffineForOp)> unrollCheckFn_ = nullptr) : unrollCheckFn(unrollCheckFn_) {};
   void runOnOperation() override {

     auto module = getOperation();
     module.walk<mlir::WalkOrder::PostOrder>([&](mlir::AffineForOp forOp) {
      if (!unrollCheckFn(forOp)) return;
      mlir::OpBuilder builder(forOp->getContext());
      forOp->setAttr(std::string("affine.loop"), builder.getStringAttr("unroll"));
     });
   }
  mlir::function_ref<bool(mlir::AffineForOp)> unrollCheckFn;
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> 
UnrollAttributePass(mlir::function_ref<bool(mlir::AffineForOp)> unrollCheckFn = nullptr) {
  return std::make_unique<UnrollAttribute>(unrollCheckFn);
}

void Rewriter::unrollAttribute(mlir::ModuleOp module, mlir::function_ref<bool(mlir::AffineForOp)> unrollCheckFn) {

  mlir::PassManager pm(module.getContext());
  pm.addPass(UnrollAttributePass(unrollCheckFn));
  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Unroll affine for failed.";
  }
  return;
}

// void Rewriter::loweringAffineDialect(mlir::ModuleOp module) {
//   mlir::PassManager pm(module.getContext());
//   pm.addPass(UnrollAttributePass(unrollCheckFn));
//   if (mlir::failed(pm.run(module))) {
//     llvm::errs() << "Unroll affine for failed.";
//   }
//   return;
// }

void Rewriter::change_double_buffer(mlir::AffineForOp scope, mlir::Value buffer) {
  scope.walk<mlir::WalkOrder::PostOrder>([&](mlir::AffineVectorLoadOp load) {
    auto mem = load.getMemref();
    if (mem == buffer) {
      auto builder = mlir::OpBuilder(load);
      auto vecT = load.getVectorType();
      auto oldMap = load.getAffineMap();
      auto operands = load.getMapOperands();
      auto oldExprs = oldMap.getResults();
      llvm::SmallVector<mlir::AffineExpr> exprs;
      for (int i = 0; i < oldExprs.size(); i++) {
        if (i == 0) {
          auto binaryExpr = oldExprs[i].dyn_cast<mlir::AffineBinaryOpExpr>();
          assert(binaryExpr && binaryExpr.getKind() == mlir::AffineExprKind::Mod);
          auto constExpr = binaryExpr.getRHS().dyn_cast<mlir::AffineConstantExpr>();
          assert(constExpr && constExpr.getValue() == 2);
          exprs.push_back((binaryExpr.getLHS() + 1) % 2);
        } else {
          exprs.push_back(oldExprs[i]);
        }
      }
      auto map = mlir::AffineMap::get(/*dimCount*/oldMap.getNumDims(), 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), load->getContext());
      auto ld = builder.create<mlir::AffineVectorLoadOp>(builder.getUnknownLoc(), vecT, buffer, map, operands);
      load.getResult().replaceAllUsesWith(ld.getResult());
      load.erase();
    }
  });
  ///TODO: support more operations for change double buffer.
  
}

void Rewriter::set_buffer(mlir::OpBuilder& builder, mlir::Value mem, mlir::Value targetValue) {
  auto type = mem.getType().dyn_cast<mlir::MemRefType>();
  auto shape = type.getShape();
  mlir::SmallVector<int64_t, 8> lowerBounds(shape.size(), /*Value=*/0);
  mlir::SmallVector<int64_t, 8> steps(shape.size(), /*Value=*/1);
  mlir::SmallVector<int64_t, 8> upperBounds(shape.begin(), shape.end());
  mlir::buildAffineLoopNest(
    builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
      nestedBuilder.create<mlir::AffineStoreOp>(nestedBuilder.getUnknownLoc(), 
        targetValue, mem, ivs);
    }
  );
}

mlir::AffineForOp Rewriter::create_constant_loop(mlir::OpBuilder& builder, int64_t lowerBound, int64_t upperBound, int64_t step) {
  auto loop_body = [&](mlir::OpBuilder &kBuilder, mlir::Location kLoc, mlir::Value iv,
                      mlir::ValueRange iterArgs) {
    kBuilder.create<mlir::AffineYieldOp>(kBuilder.getUnknownLoc());
  };
  auto loop = builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), 
    lowerBound, upperBound, step, /*iterArgs=llvm::None*/ llvm::None, loop_body);
  return loop;
}

mlir::AffineForOp Rewriter::outer_product(mlir::OpBuilder& builder, mlir::Value tileC, 
  mlir::Value fragA, mlir::Value fragB, int64_t m, int64_t n) {
  auto outerLoop = Rewriter::create_constant_loop(builder, 0, m, 1);
  auto ip = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(outerLoop.getBody());
  auto innerLoop = Rewriter::create_constant_loop(builder, 0, n, 1);
  builder.setInsertionPointToStart(innerLoop.getBody());
  {
    auto i = outerLoop.getInductionVar();
    auto j = innerLoop.getInductionVar();
    auto ld_a = builder.create<mlir::AffineLoadOp>(
      builder.getUnknownLoc(), fragA, mlir::ValueRange({i}));
    auto ld_b = builder.create<mlir::AffineLoadOp>(
      builder.getUnknownLoc(), fragB, mlir::ValueRange({j}));
    auto ld_c = builder.create<mlir::AffineLoadOp>(
      builder.getUnknownLoc(), tileC, mlir::ValueRange({i, j}));
    auto mul = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), ld_a, ld_b);
    auto add = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), mul, ld_c);
    builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), add.getResult(), tileC, mlir::ValueRange({i, j}));
     
  }
  builder.restoreInsertionPoint(ip);
}

}