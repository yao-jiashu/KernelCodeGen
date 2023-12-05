#include "Optimizer/Optimizer.h"


namespace KernelCodeGen {

std::map<std::string, int> MatmulOprimizer::matmulConfig;

struct LoadOrStoreOp {
  enum MemRSKind {
    LOAD = 0,
    STORE = 1,
  };

  LoadOrStoreOp() = default;
  LoadOrStoreOp(mlir::AffineLoadOp loadOp_) : loadOp(loadOp_), kind(LOAD) {}
  LoadOrStoreOp(mlir::AffineStoreOp storeOp_) : storeOp(storeOp_), kind(STORE) {}
  LoadOrStoreOp(const LoadOrStoreOp&  other) {
    kind = other.kind;
    kind == LoadOrStoreOp::LOAD ? loadOp = other.loadOp : storeOp = other.storeOp;
  }
  // LoadOrStoreOp& LoadOrStoreOp(LoadOrStoreOp&&  other) {
  //   kind = other.kind;
  //   kind == LoadOrStoreOp::LOAD ? loadOp = other.loadOp : storeOp = other.storeOp;
  //   return *this;
  // }
  mlir::AffineForOp getParentLoop() {
    auto* parent = kind == LoadOrStoreOp::LOAD ? loadOp->getParentOp() :
                                                storeOp->getParentOp();
    auto forOp = mlir::dyn_cast<mlir::AffineForOp>(parent);
    return forOp;
  }

  mlir::Operation::operand_range getIndexes() {
    return kind == LoadOrStoreOp::LOAD ? loadOp.getIndices(): 
                                         storeOp.getIndices();
  }

  mlir::Value getMemory() {
    return kind == LoadOrStoreOp::LOAD ? loadOp.getMemref() : 
                                         storeOp.getMemref();
  }

  mlir::AffineLoadOp loadOp;
  mlir::AffineStoreOp storeOp;
  MemRSKind kind;
};

int getLoopIndex(const std::vector<mlir::AffineForOp>& loops, mlir::AffineForOp forOp) {
  int index = -1;
  for (auto loop : loops) {
    index += 1;
    if (loop == forOp) return index;
  }
  return -1;
}

int searchInductionVar(const std::vector<mlir::AffineForOp>& loops, mlir::Value val) {
  auto ivArg = val.dyn_cast<mlir::BlockArgument>();
  if (!ivArg || !ivArg.getOwner())
    return -1;
  auto *containingInst = ivArg.getOwner()->getParent()->getParentOp();
  auto forOp = mlir::dyn_cast<mlir::AffineForOp>(containingInst);
  if (!forOp || forOp.getInductionVar() != val) return -1;

  return getLoopIndex(loops, forOp);
} 


bool MatmulOprimizer::isMatmulPattern(mlir::AffineForOp rootOp) {
  std::vector<mlir::AffineForOp> loops;
  rootOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineForOp forOp) {
    loops.push_back(forOp);
  });
  ///TODO: descirpe [ Check 1 ]. (M, N, K) 3 nested loops.
  if (loops.size() != 3) return false;
  // Reduction loop.
  if (loops[2].getIterOperands().size() == 0) return false;

  std::map<mlir::AffineForOp, std::vector<LoadOrStoreOp>, CompareLoop> scopeLdSt;

  bool result = true;

  auto collectLoadStoreOps = [&](LoadOrStoreOp& op) {
    auto parentOp = op.getParentLoop();
    auto index = getLoopIndex(loops, parentOp);
    if (index == -1) {
      result = false;
      return;
    }
    if (scopeLdSt.count(loops[index]) == 0) {
      scopeLdSt[loops[index]] = std::move(std::vector<LoadOrStoreOp>{op});
    } else {
      auto ldstVector = scopeLdSt.find(loops[index]);
      ldstVector->second.push_back(op);
    }
  };

  rootOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineLoadOp loadOp) {
    auto op = LoadOrStoreOp(loadOp);
    collectLoadStoreOps(op);
  });
  if (!result) return false;
  rootOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineStoreOp storeOp) {
    auto op = LoadOrStoreOp(storeOp);
    collectLoadStoreOps(op);
  });
  if (!result) return false;

  MemoryBuffer buf;

  ///TODO: [ Check 2 ]
  auto MNLoopScopeCheck = [&]() {
    // need to store C[i][j] in the MN scope.
    // but we can't expect it equals to 1, 
    //  as there maybe kernel fusion bringing additional memory access.
    if (scopeLdSt.count(loops[1]) == 0) {
      result = false;
      return;
    }
    bool storeCij = false;

    auto mnLdStVector = scopeLdSt.find(loops[1])->second;
    for (auto ldst : mnLdStVector) {
      auto&& indexes = ldst.getIndexes();
      std::vector<int> offsets;
      // all index to memory buffer must depend on M，N Loops.
      for (auto index : indexes) {
        int offset = searchInductionVar({loops[0], loops[1]}, index);
        if (offset == -1) {
          result = false;
          return;
        }
        offsets.push_back(offset);
      }
      
      if (!storeCij && ldst.kind == LoadOrStoreOp::STORE) {
        if (offsets == std::vector<int> {0, 1}) {
          storeCij = true;
          buf.C = ldst.getMemory();
        }
      }
    }

    if (!storeCij) {
      result = false;
    }
  };

  ///TODO: [ Check 3 ]
  auto KLoopScopeCheck = [&]() {
    //at least: read A[i][k], read B[k][j]
    if (scopeLdSt.count(loops[2])  == 0 ||
        scopeLdSt.find(loops[2])->second.size() < 2) {
      result = false;
      return;
    }
    bool readAik = false;
    bool readBkj = false;
    // bool writeCij = false;
    auto mnLdStVector = scopeLdSt.find(loops[2])->second;
    for (auto ldst : mnLdStVector) {
      auto&& indexes = ldst.getIndexes();
      std::vector<int> offsets;
      for (auto index : indexes) {
        // all index to memory buffer must depend on M，N, K Loops.
        int offset = searchInductionVar({loops[0], loops[1], loops[2]}, index);
        if (offset == -1) {
          result = false;
          return;
        }
        offsets.push_back(offset);
      }

      if (!readAik && ldst.kind == LoadOrStoreOp::LOAD) {
        if (offsets == std::vector<int> {0, 2}) {
          readAik = true;
          buf.A = ldst.getMemory();
        }
      }

      if (!readBkj && ldst.kind == LoadOrStoreOp::LOAD) {
        if (offsets == std::vector<int> {2, 1}) {
          readBkj = true;
          buf.B = ldst.getMemory();
        }
      }
      
      // if (!writeCij && ldst.kind == LoadOrStoreOp::STORE) {
      //   if (offsets == std::vector<int> {0, 1}) {
      //     writeCij = true;
      //     assert(buf.C == ldst.getMemory());
      //   }
      // }
    }

    // if (!(readAik && readBkj && writeCij)) {
    //   result = false;
    // }
    if (!(readAik && readBkj)) {
      result = false;
    }
  };

  MNLoopScopeCheck();
  KLoopScopeCheck();

  if (result) {
    matmuls.insert(rootOp);
    matmulLoops[rootOp] = loops;
    RWBuffers[rootOp] = buf;
  }
  return result;
}

bool MatmulOprimizer::applicable(mlir::ModuleOp& module) {
  clear();
  auto outermostLoops = Analyzer::collectOutermostLoop(module);
  bool res = false;
  for (auto loop : outermostLoops) {
    if (isMatmulPattern(loop)) {
      res = true;
    }
  }
  return res;
}

int64_t smAReadSride(int64_t blockDim, int64_t warpSize) {
  int64_t warpNum = blockDim / warpSize;
  int64_t laneNum = warpSize;
  //warp orgnize: 2 x 4
  std::vector<int64_t> warpOrg {2, 4};
  std::vector<int64_t> threadOrg {8, 4};
  return (warpNum / warpOrg[1]) * threadOrg[0];
}

int64_t smBReadSride(int64_t blockDim, int64_t warpSize) {
  int64_t warpNum = blockDim / warpSize;
  int64_t laneNum = warpSize;
  //warp orgnize: 2 x 4
  std::vector<int64_t> warpOrg {2, 4};
  std::vector<int64_t> threadOrg {8, 4};
  return (warpNum / warpOrg[0]) * threadOrg[1];
}

mlir::AffineMap MatmulOprimizer::getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder) {
  auto dim0 = builder.getAffineDimExpr(0);
  auto dim1 = builder.getAffineDimExpr(1);
  auto dim2 = builder.getAffineDimExpr(2);
  auto dim3 = builder.getAffineDimExpr(3);
  auto dim4 = builder.getAffineDimExpr(4);
  auto dim5 = builder.getAffineDimExpr(5);
  auto dim6 = builder.getAffineDimExpr(6);
  auto dim7 = builder.getAffineDimExpr(7);
  int64_t blockDimY = matmulConfig["BLOCK_SIZE_M"] / matmulConfig["THREAD_SIZE_M"];
  int64_t blockDimX = matmulConfig["BLOCK_SIZE_N"] / matmulConfig["THREAD_SIZE_N"];
  bool vectorize = matmulConfig.count("VECTORIZE_WIDTH") != 0;
  int width = vectorize ? matmulConfig["VECTORIZE_WIDTH"] : 1;

  std::vector<int64_t> warpOrg {2, 4};  
  std::vector<int64_t> threadOrg {8, 4};

  if (mapIdentifier == "loadTileA") {
    // dims are:[dim0, dim1, dim2, dim3, dim4]
    // operands are: [threadIdx.y, threadIdx.x, blockIdx.y, k_outer, iv]
    // iv represent a block copy for iv times. 
    auto threadIdExpr = dim0 * blockDimX + dim1;
    auto virtaulThreadIxExpr = threadIdExpr + dim4 * blockDimY * blockDimX;
    auto M_Offset = virtaulThreadIxExpr.floorDiv(static_cast<uint64_t>(matmulConfig["BLOCK_SIZE_K"]) / width);
    auto K_Offset = virtaulThreadIxExpr % (static_cast<uint64_t>(matmulConfig["BLOCK_SIZE_K"]) / width); 
    auto M_Base = dim2 * matmulConfig["BLOCK_SIZE_M"];
    auto K_Base = dim3;
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(M_Offset + M_Base);
    exprs.push_back(K_Offset * width + K_Base);
    return mlir::AffineMap::get(/*dimCount*/5, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "loadTileB") {
    // dims are:[dim0, dim1, dim2, dim3, dim4]
    // operands are: [threadIdx.y, threadIdx.x, k_outer, blockIdx.x, iv]
    auto threadIdExpr = dim0 * blockDimX + dim1;
    auto virtaulThreadIxExpr = threadIdExpr + dim4 * blockDimY * blockDimX;
    auto K_Offset = virtaulThreadIxExpr.floorDiv(static_cast<uint64_t>(matmulConfig["BLOCK_SIZE_N"]) / width);
    auto N_Offset = virtaulThreadIxExpr % (static_cast<uint64_t>(matmulConfig["BLOCK_SIZE_N"]) / width); 
    auto K_Base = dim2;
    auto N_Base = dim3 * matmulConfig["BLOCK_SIZE_N"];
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(K_Offset + K_Base);
    exprs.push_back(N_Offset * width + N_Base);
    return mlir::AffineMap::get(/*dimCount*/5, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "storeTileA") {
    // dims are:[dim0, dim1, dim2, dim3]
    // operands are: [threadIdx.y, threadIdx.x, iv, ivInVector]
    auto threadIdExpr = dim0 * blockDimX + dim1;
    auto virtaulThreadIxExpr = threadIdExpr + dim2 * blockDimY * blockDimX;
    auto M_Offset = virtaulThreadIxExpr.floorDiv(static_cast<uint64_t>(matmulConfig["BLOCK_SIZE_K"]) / width);
    auto K_Offset = virtaulThreadIxExpr % (static_cast<uint64_t>(matmulConfig["BLOCK_SIZE_K"]) / width);
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(K_Offset * width + dim3);
    exprs.push_back(M_Offset);
    return mlir::AffineMap::get(/*dimCount*/4, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "storeTileB") {
    // dims are:[dim0, dim1, dim2]
    // operands are: [threadIdx.y, threadIdx.x, iv]
    auto threadIdExpr = dim0 * blockDimX + dim1;
    auto virtaulThreadIxExpr = threadIdExpr + dim2 * blockDimY * blockDimX;
    auto K_Offset = virtaulThreadIxExpr.floorDiv(static_cast<uint64_t>(matmulConfig["BLOCK_SIZE_N"]) / width);
    auto N_Offset = virtaulThreadIxExpr % (static_cast<uint64_t>(matmulConfig["BLOCK_SIZE_N"]) / width); 
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(K_Offset);
    exprs.push_back(N_Offset * width);
    return mlir::AffineMap::get(/*dimCount*/3, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "loadFragA") {
    // dims are:[dim0, dim1, dim2, dim3]
    // operands are: [threadIdx.y, threadIdx.x, k_inner, iv]
    auto threadIdExpr = dim0 * blockDimX + dim1;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(matmulConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(matmulConfig["WARP_SIZE"]);

    auto M_offset = laneId.floorDiv(threadOrg[1]) + threadOrg[0] * (warpId.floorDiv(warpOrg[1]) + dim3 * warpOrg[0]);
    auto K_offset = dim2;
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(K_offset);
    exprs.push_back(M_offset * width);
    return mlir::AffineMap::get(/*dimCount*/4, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "loadFragB") {
    // dims are:[dim0, dim1, dim2, dim3]
    // operands are: [threadIdx.y, threadIdx.x, k_inner, iv]
    auto threadIdExpr = dim0 * blockDimX + dim1;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(matmulConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(matmulConfig["WARP_SIZE"]);

    auto N_offset = laneId % threadOrg[1] + threadOrg[1] * (warpId % warpOrg[1] + dim3 * warpOrg[1]);
    auto K_offset = dim2;
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(K_offset);
    exprs.push_back(N_offset * width);
    return mlir::AffineMap::get(/*dimCount*/4, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "cacheReadA" || mapIdentifier == "cacheReadB") {
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0);
    return mlir::AffineMap::get(/*dimCount*/1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "cacheWriteC") {
    // dims are:[dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7]
    // operands are: [threadIdx.y, threadIdx.x, blockIdx.y, blockIdx.x, iv0, iv1, iv2, iv3]
    auto M_index = dim6 + dim4 * blockDimY + dim0 * width + dim2 * matmulConfig["BLOCK_SIZE_M"];
    auto N_index = dim7 + dim5 * blockDimX + dim1 * width + dim3 * matmulConfig["BLOCK_SIZE_N"];
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(M_index);
    exprs.push_back(N_index);
    return mlir::AffineMap::get(/*dimCount*/8, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else {
    assert(false);
  }
}

void MatmulOprimizer::applyOptimzer(mlir::ModuleOp& module, mlir::OpBuilder& builder) {

  for (auto matmul : matmuls) {
    auto loops = matmulLoops[matmul];
    auto loopM = loops[0], loopN = loops[1], loopK = loops[2];
    auto buffers = RWBuffers[matmul];
    auto A = buffers.A, B = buffers.B, C = buffers.C;
    
    auto m_axes = Rewriter::split(loopM, 3, {matmulConfig["THREAD_SIZE_M"], matmulConfig["BLOCK_SIZE_M"]});
    auto n_axes = Rewriter::split(loopN, 3, {matmulConfig["THREAD_SIZE_N"], matmulConfig["BLOCK_SIZE_N"]});

    module.dump();

    auto m_outer = m_axes[0], m_mider = m_axes[1], m_inner = m_axes[2];
    auto n_outer = n_axes[0], n_mider = n_axes[1], n_inner = n_axes[2];


    Rewriter::reorder({m_outer, n_outer, m_mider, n_mider, m_inner, n_inner});
    module.dump();

    auto gridLevel = Rewriter::parallel({m_outer, n_outer});
    auto blockLevel = Rewriter::parallel({m_mider, n_mider});
    module.dump();


    std::vector<mlir::AffineForOp> kmn_axes{loopK, m_inner, n_inner};
    auto tileC = Rewriter::bufferizeLoopCarryVar(kmn_axes);
    loopK = kmn_axes[0], m_inner = kmn_axes[1], n_inner = kmn_axes[2];
    module.dump();

    Rewriter::reorder({loopK, m_inner, n_inner});
    module.dump();

    auto k_axes = Rewriter::split(loopK, 2, {matmulConfig["BLOCK_SIZE_K"]});
    auto k_outer = k_axes[0], k_inner = k_axes[1];
    module.dump();

    int64_t blockThreads;
    auto blockDim = Rewriter::getParallelNumber(blockLevel, blockThreads);

    auto ldgASize = matmulConfig["BLOCK_SIZE_K"] * matmulConfig["BLOCK_SIZE_M"] / blockThreads;
    auto ldgBSize = matmulConfig["BLOCK_SIZE_K"] * matmulConfig["BLOCK_SIZE_N"] / blockThreads;
    auto fragASize = matmulConfig["BLOCK_SIZE_M"] / smAReadSride(blockThreads, matmulConfig["WARP_SIZE"]);
    auto fragBSize = matmulConfig["BLOCK_SIZE_N"] / smBReadSride(blockThreads, matmulConfig["WARP_SIZE"]);
    auto elementA = A.getType().dyn_cast<mlir::MemRefType>().getElementType();
    auto elementB = B.getType().dyn_cast<mlir::MemRefType>().getElementType();

    auto fragB = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {fragBSize}, elementB);
    auto fragA = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {fragASize}, elementA);

    auto tileB = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {ldgBSize}, elementB);
    auto tileA = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {ldgASize}, elementA);
    auto smB = Rewriter::alloc_buffer(/*parallelLevel*/gridLevel, MemorySpace::shared,
            {matmulConfig["BLOCK_SIZE_K"], matmulConfig["BLOCK_SIZE_N"]}, elementB);
    auto smA = Rewriter::alloc_buffer(/*parallelLevel*/gridLevel, MemorySpace::shared,
            {matmulConfig["BLOCK_SIZE_K"], matmulConfig["BLOCK_SIZE_M"]}, elementA);
    module.dump();
    
    auto blockIdx = Rewriter::getParallelIdx(gridLevel);
    auto threadIdx = Rewriter::getParallelIdx(blockLevel);
    
    auto loadTileAMap = getAffineMap("loadTileA", builder);
    auto loadTileA = Rewriter::read(A, tileA, loadTileAMap, {threadIdx[0], threadIdx[1], blockIdx[0], k_outer.getInductionVar()}, 
                      matmulConfig["VECTORIZE_WIDTH"], k_outer, Position::begin);
    auto loadTileBMap = getAffineMap("loadTileB", builder);
    auto loadTileB = Rewriter::read(B, tileB, loadTileBMap, 
                      {threadIdx[0], threadIdx[1], k_outer.getInductionVar(), blockIdx[1]}, 
                      matmulConfig["VECTORIZE_WIDTH"], loadTileA, Position::after);
    module.dump();

    auto storeTileAMap = getAffineMap("storeTileA", builder);
    auto storeTileA = Rewriter::write(tileA, smA, storeTileAMap, {threadIdx[0], threadIdx[1]}, 
                        matmulConfig["VECTORIZE_WIDTH"], loadTileB, Position::after);
    auto storeTileBMap = getAffineMap("storeTileB", builder);
    auto storeTileB = Rewriter::write(tileB, smB, storeTileBMap, {threadIdx[0], threadIdx[1]}, 
                                matmulConfig["VECTORIZE_WIDTH"], storeTileA, Position::after);
    auto gpuBarrierPrefix = Rewriter::barrier(loadTileA, Position::before);
    auto gpuBarrierSuffix = Rewriter::barrier(storeTileB, Position::after);

    module.dump();

    auto loadFragAMap = getAffineMap("loadFragA", builder);
    auto loadFragA = Rewriter::read(smA, fragA, loadFragAMap, {threadIdx[0], threadIdx[1], k_inner.getInductionVar()}, 
                      matmulConfig["VECTORIZE_WIDTH"], k_inner, Position::begin);
    auto loadFragBMap = getAffineMap("loadFragB", builder);
    auto loadFragB = Rewriter::read(smB, fragB, loadFragBMap, {threadIdx[0], threadIdx[1], k_inner.getInductionVar()}, 
                      matmulConfig["VECTORIZE_WIDTH"], loadFragA, Position::after);
    module.dump();

    Rewriter::cache_read(k_inner, A, fragA, getAffineMap("cacheReadA", builder), {m_inner.getInductionVar()});
    Rewriter::cache_read(k_inner, B, fragB, getAffineMap("cacheReadB", builder), {n_inner.getInductionVar()});
    module.dump();

    auto writeCbody = Rewriter::get_write(blockLevel, C);
    assert(writeCbody.size() == 1);
    auto m_inner_axes = Rewriter::split(writeCbody[0][0], 2, {matmulConfig["VECTORIZE_WIDTH"]});
    auto n_inner_axes = Rewriter::split(writeCbody[0][1], 2, {matmulConfig["VECTORIZE_WIDTH"]});
    auto m_inner_0 = m_inner_axes[0], m_inner_1 = m_inner_axes[1];
    auto n_inner_0 = n_inner_axes[0], n_inner_1 = n_inner_axes[1];
    Rewriter::reorder({m_inner_0, n_inner_0, m_inner_1, n_inner_1});
    module.dump();

    Rewriter::cache_write(m_inner_0, C, tileC, getAffineMap("cacheWriteC", builder), 
                          {threadIdx[0], threadIdx[1], blockIdx[0], blockIdx[1], m_inner_0.getInductionVar(),
                          n_inner_0.getInductionVar(), m_inner_1.getInductionVar(), n_inner_1.getInductionVar()});
    module.dump();

    Rewriter::vectorize(n_inner_1, matmulConfig["VECTORIZE_WIDTH"]);
    module.dump();

    auto doubleLoadTileB = Rewriter::pipeline({loadTileB, storeTileB}, smB, k_outer);
    auto doubleLoadTileA = Rewriter::pipeline({loadTileA, storeTileA}, smA, k_outer);
    auto doubleLoadFragB = Rewriter::pipeline({loadFragB}, fragB, k_inner);
    auto doubleLoadFragA = Rewriter::pipeline({loadFragA}, fragA, k_inner);
    module.dump();

    Rewriter::detach_last_loop(k_inner);
    module.dump();

    Rewriter::schedule(doubleLoadTileA[0][0], doubleLoadTileB[0][0], Position::before);
    Rewriter::schedule(doubleLoadTileA[0][1], doubleLoadTileB[0][1], Position::before); 
    Rewriter::schedule(gpuBarrierPrefix, doubleLoadTileB[0][1], Position::after);
    Rewriter::schedule(doubleLoadTileB[1][0], doubleLoadTileA[1][0], Position::after);
    Rewriter::schedule(doubleLoadTileA[1][1], doubleLoadTileB[1][1], Position::before);
    Rewriter::schedule(gpuBarrierSuffix, doubleLoadTileB[1][1], Position::after);
    auto ifOp = doubleLoadTileA[1][1]->getParentOp();
    Rewriter::schedule(ifOp, k_inner, Position::after); 
    Rewriter::extract_loop(doubleLoadFragA[0][0], k_outer, /*iteration*/0);
    Rewriter::extract_loop(doubleLoadFragB[0][0], k_outer, /*iteration*/0);
    Rewriter::schedule(doubleLoadFragB[0][0], k_outer, Position::end);
    Rewriter::schedule(doubleLoadFragA[0][0], k_outer, Position::end);
    module.dump();

    Rewriter::take_off_true_if(module);
    Rewriter::delete_false_if(module);
    module.dump();

    int64_t threshold = std::max(matmulConfig["BLOCK_SIZE_K"], std::max(matmulConfig["THREAD_SIZE_M"], matmulConfig["THREAD_SIZE_N"]));
    Rewriter::unroll(module, [&](mlir::AffineForOp forOp)->bool {
      if (!forOp.hasConstantBounds()) return false;
      auto step = forOp.getStep();
      auto ub = forOp.getConstantUpperBound();
      auto lb = forOp.getConstantLowerBound();
      auto times = (ub - lb) / step;
      if (times >= std::min<int64_t>(threshold, matmulConfig["VECTORIZE_WIDTH"])) return false;
      return true;
    });
    module.dump();

    Rewriter::unrollAttribute(module, [&](mlir::AffineForOp forOp)->bool {
      if (!forOp.hasConstantBounds()) return false;
      auto step = forOp.getStep();
      auto ub = forOp.getConstantUpperBound();
      auto lb = forOp.getConstantLowerBound();
      auto times = (ub - lb) / step;
      if (times > threshold) return false;
      return true;
    });
    module.dump();

  }
    
}

}