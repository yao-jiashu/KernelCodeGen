#include "Optimizer/Optimizer.h"
#include "log.h"
#include <cfloat>

#define DUMP(module)                    \
{                                       \
  if (KCGLog::level == Log::Debug) {    \
    module.dump();                      \
  }                                     \
}                                       

namespace KernelCodeGen {

std::map<std::string, int> MatmulOptimizer::matmulConfig;
std::map<std::string, int> FMHAOptimizer::fmhaConfig;

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


// bool MatmulOptimizer::isMatmulPattern(mlir::AffineForOp rootOp) {
//   std::vector<mlir::AffineForOp> loops;
//   rootOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineForOp forOp) {
//     loops.push_back(forOp);
//   });
//   ///TODO: descirpe [ Check 1 ]. (M, N, K) 3 nested loops.
//   if (loops.size() != 3) return false;
//   // Reduction loop.
//   if (loops[2].getIterOperands().size() == 0) return false;

//   std::map<mlir::AffineForOp, std::vector<LoadOrStoreOp>, CompareLoop> scopeLdSt;

//   bool result = true;

//   auto collectLoadStoreOps = [&](LoadOrStoreOp& op) {
//     auto parentOp = op.getParentLoop();
//     auto index = getLoopIndex(loops, parentOp);
//     if (index == -1) {
//       result = false;
//       return;
//     }
//     if (scopeLdSt.count(loops[index]) == 0) {
//       scopeLdSt[loops[index]] = std::move(std::vector<LoadOrStoreOp>{op});
//     } else {
//       auto ldstVector = scopeLdSt.find(loops[index]);
//       ldstVector->second.push_back(op);
//     }
//   };

//   rootOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineLoadOp loadOp) {
//     auto op = LoadOrStoreOp(loadOp);
//     collectLoadStoreOps(op);
//   });
//   if (!result) return false;
//   rootOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::AffineStoreOp storeOp) {
//     auto op = LoadOrStoreOp(storeOp);
//     collectLoadStoreOps(op);
//   });
//   if (!result) return false;

//   MemoryBuffer buf;

//   ///TODO: [ Check 2 ]
//   auto MNLoopScopeCheck = [&]() {
//     // need to store C[i][j] in the MN scope.
//     // but we can't expect it equals to 1, 
//     //  as there maybe kernel fusion bringing additional memory access.
//     if (scopeLdSt.count(loops[1]) == 0) {
//       result = false;
//       return;
//     }
//     bool storeCij = false;

//     auto mnLdStVector = scopeLdSt.find(loops[1])->second;
//     for (auto ldst : mnLdStVector) {
//       auto&& indexes = ldst.getIndexes();
//       std::vector<int> offsets;
//       // all index to memory buffer must depend on M，N Loops.
//       for (auto index : indexes) {
//         int offset = searchInductionVar({loops[0], loops[1]}, index);
//         if (offset == -1) {
//           result = false;
//           return;
//         }
//         offsets.push_back(offset);
//       }
      
//       if (!storeCij && ldst.kind == LoadOrStoreOp::STORE) {
//         if (offsets == std::vector<int> {0, 1}) {
//           storeCij = true;
//           buf.C = ldst.getMemory();
//         }
//       }
//     }

//     if (!storeCij) {
//       result = false;
//     }
//   };

//   ///TODO: [ Check 3 ]
//   auto KLoopScopeCheck = [&]() {
//     //at least: read A[i][k], read B[k][j]
//     if (scopeLdSt.count(loops[2])  == 0 ||
//         scopeLdSt.find(loops[2])->second.size() < 2) {
//       result = false;
//       return;
//     }
//     bool readAik = false;
//     bool readBkj = false;
//     // bool writeCij = false;
//     auto mnLdStVector = scopeLdSt.find(loops[2])->second;
//     for (auto ldst : mnLdStVector) {
//       auto&& indexes = ldst.getIndexes();
//       std::vector<int> offsets;
//       for (auto index : indexes) {
//         // all index to memory buffer must depend on M，N, K Loops.
//         int offset = searchInductionVar({loops[0], loops[1], loops[2]}, index);
//         if (offset == -1) {
//           result = false;
//           return;
//         }
//         offsets.push_back(offset);
//       }

//       if (!readAik && ldst.kind == LoadOrStoreOp::LOAD) {
//         if (offsets == std::vector<int> {0, 2}) {
//           readAik = true;
//           buf.A = ldst.getMemory();
//         }
//       }

//       if (!readBkj && ldst.kind == LoadOrStoreOp::LOAD) {
//         if (offsets == std::vector<int> {2, 1}) {
//           readBkj = true;
//           buf.B = ldst.getMemory();
//         }
//       }
      
//       // if (!writeCij && ldst.kind == LoadOrStoreOp::STORE) {
//       //   if (offsets == std::vector<int> {0, 1}) {
//       //     writeCij = true;
//       //     assert(buf.C == ldst.getMemory());
//       //   }
//       // }
//     }

//     // if (!(readAik && readBkj && writeCij)) {
//     //   result = false;
//     // }
//     if (!(readAik && readBkj)) {
//       result = false;
//     }
//   };

//   MNLoopScopeCheck();
//   KLoopScopeCheck();

//   if (result) {
//     matmuls.insert(rootOp);
//     matmulLoops[rootOp] = loops;
//     matmulBuffers[rootOp] = buf;
//   }
//   return result;
// }

bool MatmulOptimizer::applicable(mlir::ModuleOp& module) {
  clear();
  auto&& matmulFuncs = Analyzer::collectFunctions(module, "Matmul");
  bool res = matmulFuncs.size() != 0 ? true : false;

  for (auto& matmulFunc : matmulFuncs) {
    if (matmuls.count(matmulFunc) != 0 || matmulLoops.count(matmulFunc) != 0
      || matmulBuffers.count(matmulFunc) != 0) {
      llvm::errs() << "Duplicated Matmul in module\n";
    }
    matmuls.insert(matmulFunc);
    auto&& loops = Analyzer::collectFuncLoops(matmulFunc);
    matmulLoops[matmulFunc] = std::move(loops);
    auto funcArgs = matmulFunc.front().getArguments();
    // matmulBuffers[matmulFunc] = MemoryBuffer(mlir::dyn_cast<mlir::Value>(funcArgs[0]), 
    //   mlir::dyn_cast<mlir::Value>(funcArgs[1]), mlir::dyn_cast<mlir::Value>(funcArgs[2]));
    MemoryBuffer ABC;
    ABC.A = funcArgs[0];
    ABC.B = funcArgs[1];
    ABC.C = funcArgs[2];
    // matmulBuffers[matmulFunc] = MemoryBuffer(funcArgs[0].dyn_cast<mlir::Value>(), 
    //   funcArgs[1].dyn_cast<mlir::Value>(), funcArgs[2].dyn_cast<mlir::Value>());
    matmulBuffers[matmulFunc] = ABC;
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

mlir::AffineMap MatmulOptimizer::getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder) {
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
    auto threadIdExpr = dim0 * blockDimX + dim1;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(matmulConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(matmulConfig["WARP_SIZE"]);

    auto M_offset = laneId.floorDiv(threadOrg[1]) + threadOrg[0] * (warpId.floorDiv(warpOrg[1]) + dim4.floorDiv(width) * warpOrg[0]);
    auto N_offset = laneId % threadOrg[1] + threadOrg[1] * (warpId % warpOrg[1] + dim5.floorDiv(width) * warpOrg[1]);
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim2 * matmulConfig["BLOCK_SIZE_M"] + M_offset * width + dim6);
    exprs.push_back(dim3 * matmulConfig["BLOCK_SIZE_N"] + N_offset * width + dim7);
    return mlir::AffineMap::get(/*dimCount*/8, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else {
    assert(false);
  }
}

void MatmulOptimizer::applyOptimzer(mlir::ModuleOp& module, mlir::OpBuilder& builder) {

  for (auto& matmul : matmuls) {
    matmul->setAttr(std::string("func.state"), builder.getStringAttr("gpu"));
    auto loops = matmulLoops[matmul];
    auto loopM = loops[0], loopN = loops[1], loopK = loops[2];
    auto buffers = matmulBuffers[matmul];
    auto A = buffers.A, B = buffers.B, C = buffers.C;
    
    auto m_axes = Rewriter::split(loopM, 3, {matmulConfig["THREAD_SIZE_M"], matmulConfig["BLOCK_SIZE_M"]});
    auto n_axes = Rewriter::split(loopN, 3, {matmulConfig["THREAD_SIZE_N"], matmulConfig["BLOCK_SIZE_N"]});

    DUMP(module);

    auto m_outer = m_axes[0], m_mider = m_axes[1], m_inner = m_axes[2];
    auto n_outer = n_axes[0], n_mider = n_axes[1], n_inner = n_axes[2];


    Rewriter::reorder({m_outer, n_outer, m_mider, n_mider, m_inner, n_inner});
    DUMP(module);

    auto gridLevel = Rewriter::parallel({m_outer, n_outer});
    auto blockLevel = Rewriter::parallel({m_mider, n_mider});
    DUMP(module);


    std::vector<mlir::AffineForOp> kmn_axes{loopK, m_inner, n_inner};
    auto tileC = Rewriter::bufferizeLoopCarryVar(kmn_axes);
    loopK = kmn_axes[0], m_inner = kmn_axes[1], n_inner = kmn_axes[2];
    DUMP(module);

    Rewriter::reorder({loopK, m_inner, n_inner});
    DUMP(module);

    auto k_axes = Rewriter::split(loopK, 2, {matmulConfig["BLOCK_SIZE_K"]});
    auto k_outer = k_axes[0], k_inner = k_axes[1];
    DUMP(module);

    int64_t blockThreads;
    auto blockDim = Analyzer::getParallelNumber(blockLevel, blockThreads);

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
    DUMP(module);
    
    auto blockIdx = Rewriter::getParallelIdx(gridLevel);
    auto threadIdx = Rewriter::getParallelIdx(blockLevel);
    
    auto loadTileAMap = getAffineMap("loadTileA", builder);
    auto loadTileA = Rewriter::read(A, tileA, loadTileAMap, {threadIdx[0], threadIdx[1], blockIdx[0], k_outer.getInductionVar()}, 
                      matmulConfig["VECTORIZE_WIDTH"], k_outer, Position::begin);
    auto loadTileBMap = getAffineMap("loadTileB", builder);
    auto loadTileB = Rewriter::read(B, tileB, loadTileBMap, 
                      {threadIdx[0], threadIdx[1], k_outer.getInductionVar(), blockIdx[1]}, 
                      matmulConfig["VECTORIZE_WIDTH"], loadTileA, Position::after);
    DUMP(module);

    auto storeTileAMap = getAffineMap("storeTileA", builder);
    auto storeTileA = Rewriter::write(tileA, smA, storeTileAMap, {threadIdx[0], threadIdx[1]}, 
                        matmulConfig["VECTORIZE_WIDTH"], loadTileB, Position::after);
    auto storeTileBMap = getAffineMap("storeTileB", builder);
    auto storeTileB = Rewriter::write(tileB, smB, storeTileBMap, {threadIdx[0], threadIdx[1]}, 
                                matmulConfig["VECTORIZE_WIDTH"], storeTileA, Position::after);
    auto gpuBarrierPrefix = Rewriter::barrier(loadTileA, Position::before);
    auto gpuBarrierSuffix = Rewriter::barrier(storeTileB, Position::after);

    DUMP(module);

    auto loadFragAMap = getAffineMap("loadFragA", builder);
    auto loadFragA = Rewriter::read(smA, fragA, loadFragAMap, {threadIdx[0], threadIdx[1], k_inner.getInductionVar()}, 
                      matmulConfig["VECTORIZE_WIDTH"], k_inner, Position::begin);
    auto loadFragBMap = getAffineMap("loadFragB", builder);
    auto loadFragB = Rewriter::read(smB, fragB, loadFragBMap, {threadIdx[0], threadIdx[1], k_inner.getInductionVar()}, 
                      matmulConfig["VECTORIZE_WIDTH"], loadFragA, Position::after);
    DUMP(module);

    Rewriter::cache_read(k_inner, A, fragA, getAffineMap("cacheReadA", builder), {m_inner.getInductionVar()});
    Rewriter::cache_read(k_inner, B, fragB, getAffineMap("cacheReadB", builder), {n_inner.getInductionVar()});
    DUMP(module);

    auto writeCbody = Rewriter::get_write(blockLevel, C);
    assert(writeCbody.size() == 1);
    auto m_inner_axes = Rewriter::split(writeCbody[0][0], 2, {matmulConfig["VECTORIZE_WIDTH"]});
    auto n_inner_axes = Rewriter::split(writeCbody[0][1], 2, {matmulConfig["VECTORIZE_WIDTH"]});
    auto m_inner_0 = m_inner_axes[0], m_inner_1 = m_inner_axes[1];
    auto n_inner_0 = n_inner_axes[0], n_inner_1 = n_inner_axes[1];
    Rewriter::reorder({m_inner_0, n_inner_0, m_inner_1, n_inner_1});
    DUMP(module);

    Rewriter::cache_write(m_inner_0, C, tileC, getAffineMap("cacheWriteC", builder), 
                          {threadIdx[0], threadIdx[1], blockIdx[0], blockIdx[1], m_inner_0.getInductionVar(),
                          n_inner_0.getInductionVar(), m_inner_1.getInductionVar(), n_inner_1.getInductionVar()});
    DUMP(module);

    Rewriter::vectorize(n_inner_1, matmulConfig["VECTORIZE_WIDTH"]);
    DUMP(module);
    
    auto doubleLoadTileB = Rewriter::pipeline({loadTileB, storeTileB}, smB, k_outer);
    auto doubleLoadTileA = Rewriter::pipeline({loadTileA, storeTileA}, smA, k_outer);
    auto doubleLoadFragB = Rewriter::pipeline({loadFragB}, fragB, k_inner);
    auto doubleLoadFragA = Rewriter::pipeline({loadFragA}, fragA, k_inner);
    DUMP(module);

    Rewriter::detach_last_loop(k_inner);
    DUMP(module);

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
    DUMP(module);

    Rewriter::change_double_buffer(doubleLoadFragA[0][0], smA);
    Rewriter::change_double_buffer(doubleLoadFragB[0][0], smB);;
    DUMP(module);

    Rewriter::take_off_true_if(module);
    Rewriter::delete_false_if(module);
    DUMP(module);

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
    DUMP(module);

    Rewriter::unrollAttribute(module, [&](mlir::AffineForOp forOp)->bool {
      if (!forOp.hasConstantBounds()) return false;
      auto step = forOp.getStep();
      auto ub = forOp.getConstantUpperBound();
      auto lb = forOp.getConstantLowerBound();
      auto times = (ub - lb) / step;
      if (times > threshold) return false;
      return true;
    });
    DUMP(module);
  }
}

void splitString(const std::string& input, char target, std::vector<std::string>& output) {
  std::string cur {""};
  int len = input.size();
  for (int i = 0; i < len; i ++) {
    if (input[i] == target) {
      output.push_back(cur);
      cur = std::string("");
      continue;
    }
    cur += input[i];
  }
  output.push_back(cur);
}

void identifyBatchMatmul(const std::string& name, BatchMatmulDescriptor& matmul) {
  auto len = name.size();
  matmul.transA = name[len - 2] == 'N' ? false : true;
  matmul.transB = name[len - 1] == 'N' ? false : true;

  std::vector<std::string> stringFrags;
  splitString(name, '_', stringFrags);
  std::vector<int> batch;

  for (int i = 1; i < stringFrags.size() - 4; i ++) {
    batch.push_back(std::stoi(stringFrags[i]));
  }
  matmul.batch = batch;
  std::string M = stringFrags[stringFrags.size() - 4];
  std::string N = stringFrags[stringFrags.size() - 3];
  std::string K = stringFrags[stringFrags.size() - 2];
  M.erase(M.begin()); N.erase(N.begin()); K.erase(K.begin());
  matmul.m = std::stoi(M);
  matmul.n = std::stoi(N);
  matmul.k = std::stoi(K);
}

bool FMHAOptimizer::applicable(mlir::ModuleOp& module) {
  clear();
  auto funcCalls = Analyzer::collectFuncCalls(module);
  int funcNum = funcCalls.size();
  for (int i = 0; i < funcNum; i += 1) {
    auto call2Matmul = funcCalls[i];
    // auto func = call.getCalleeAttrName().str();
    auto funcName = call2Matmul.getCallee().str();
    if (funcName.find(std::string("BatchMatmul")) != std::string::npos) {
      auto matmul = Analyzer::getTargetFunction(module, funcName);
      auto attr = matmul->getAttr(std::string("func.state")).dyn_cast<mlir::StringAttr>();
      if (attr.str() != std::string("cpu")) continue;
      auto retValue = call2Matmul.getResult(0);
      auto users = retValue.getUsers();
      ///< If we want to fuse kernels, The only user has to be gurantee. 
      if (Analyzer::getUsersNumber(users) != 1) {
        continue;
      }
      auto call2Softmax = mlir::dyn_cast<mlir::func::CallOp>(*users.begin());
      if (!call2Softmax) {
        continue;
      }
      auto funcName = call2Softmax.getCallee().str();
      if (funcName.find(std::string("Softmax")) != std::string::npos) {
        auto softmax = Analyzer::getTargetFunction(module, funcName);
        auto attr = softmax->getAttr(std::string("func.state")).dyn_cast<mlir::StringAttr>();
        if (attr.str() != std::string("cpu")) continue;
        auto retValue = call2Softmax->getResult(0);
        auto users = retValue.getUsers();
        ///< If we want to fuse kernels, The only user has to be gurantee. 
        if (Analyzer::getUsersNumber(users) != 1) {
          continue;
        }
        auto call2Matmul2 = mlir::dyn_cast<mlir::func::CallOp>(*users.begin());
        if (!call2Matmul2) {
          continue;
        }
        auto funcName = call2Matmul2.getCallee().str();
        if (funcName.find(std::string("BatchMatmul")) != std::string::npos
          && uniqueFuncCalls.count(call2Matmul) == 0 && uniqueFuncCalls.count(call2Softmax) == 0
          && uniqueFuncCalls.count(call2Matmul2) == 0 && call2callsMap.count(call2Matmul) == 0
          && call2bufferMap.count(call2Matmul) == 0) {

          auto matmul2 = Analyzer::getTargetFunction(module, funcName);
          auto attr = matmul2->getAttr(std::string("func.state")).dyn_cast<mlir::StringAttr>();
          if (attr.str() != std::string("cpu")) continue;

          ///< Collect necessary information.
          MemoryBuffer buf;
          auto matmulArgs = call2Matmul.getArgOperands();
          buf.Q = matmulArgs[0];
          buf.K = matmulArgs[1];
          buf.S = matmulArgs[2];
          BatchMatmulDescriptor descMatmul;
          identifyBatchMatmul(call2Matmul.getCallee().str(), descMatmul);
          
          auto matmul2Args = call2Matmul2.getArgOperands();
          buf.V = matmul2Args[1];
          buf.O = matmul2Args[2];
          BatchMatmulDescriptor descMatmul2;
          identifyBatchMatmul(call2Matmul2.getCallee().str(), descMatmul2);
          

          ///< Not impatible.
          if (descMatmul.batch != descMatmul2.batch ||
            descMatmul.m != descMatmul2.m || descMatmul.n != descMatmul2.k) {
            continue;
          }
          buf.matmul1 = descMatmul;
          buf.matmul2 = descMatmul2;
          // descMatmul.log(); descMatmul2.log();

          ///< Record the buffer.
          call2bufferMap[call2Matmul] = buf;

          ///< Record the graph.
          call2callsMap[call2Matmul] = std::vector<mlir::func::CallOp>{call2Softmax, call2Matmul2};

          ///< Unique all three func calls.
          uniqueFuncCalls.insert(call2Matmul);
          uniqueFuncCalls.insert(call2Softmax);
          uniqueFuncCalls.insert(call2Matmul2);
        }
      }
    }
  }
}

mlir::AffineMap FMHAOptimizer::getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder) {
  auto dim0 = builder.getAffineDimExpr(0);
  auto dim1 = builder.getAffineDimExpr(1);
  auto dim2 = builder.getAffineDimExpr(2);
  auto dim3 = builder.getAffineDimExpr(3);
  auto dim4 = builder.getAffineDimExpr(4);
  auto dim5 = builder.getAffineDimExpr(5);
  auto dim6 = builder.getAffineDimExpr(6);
  auto dim7 = builder.getAffineDimExpr(7);
  int64_t BLOCK_SIZE = fmhaConfig["BLOCK_SIZE"];
  int width = fmhaConfig["Width"];

  // std::vector<int64_t> warpOrg {2, 4};
  // std::vector<int64_t> threadOrg {8, 4};

  const int LaneX_S = fmhaConfig["Bc"] / fmhaConfig["BcTileS"];
  const int LaneY_S = fmhaConfig["WARP_SIZE"] / LaneX_S;

  const int LaneX_O = fmhaConfig["Hd"] / fmhaConfig["HdTileO"] / fmhaConfig["WarpX_O"];
  const int LaneY_O = fmhaConfig["WARP_SIZE"] / LaneX_O;

  if (mapIdentifier == "loadTileQ") {
    // dims are:[dim0, dim1, dim2, dim3, dim4, dim5]
    // operands are: [blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.x, hd_outer, iv]
    // iv represent a block copy for iv times. 
    auto threadIdExpr = dim3;
    auto vThreadIdExpr = threadIdExpr + dim5 * BLOCK_SIZE;
    auto Br_Offset = vThreadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["Slice"]) / width);
    auto Hd_Offset = vThreadIdExpr % (static_cast<uint64_t>(fmhaConfig["Slice"]) / width); 
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0);
    exprs.push_back(dim1);
    exprs.push_back(dim2 * fmhaConfig["Br"] + Br_Offset);
    exprs.push_back(dim4 + Hd_Offset * width);
    return mlir::AffineMap::get(/*dimCount*/6, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "loadTileK") {
    // dims are:[dim0, dim1, dim2, dim3, dim4, dim5]
    // operands are: [blockIdx.z, blockIdx.y, threadIdx.x, k_outer, hd_outer, iv]
    // iv represent a block copy for iv times. 
    auto threadIdExpr = dim2;
    auto vThreadIdExpr = threadIdExpr + dim5 * BLOCK_SIZE;
    auto Bc_Offset = vThreadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["Slice"]) / width);
    auto Hd_Offset = vThreadIdExpr % (static_cast<uint64_t>(fmhaConfig["Slice"]) / width); 
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0);
    exprs.push_back(dim1);
    exprs.push_back(dim3 + Bc_Offset);
    exprs.push_back(dim4 + Hd_Offset * width);
    return mlir::AffineMap::get(/*dimCount*/6, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "storeTileQ") {
    // dims are:[dim0, dim1, dim2]
    // operands are: [threadIdx.x, iv, ivInVector]
    auto threadIdExpr = dim0;
    auto vThreadIdExpr = threadIdExpr + dim1 * BLOCK_SIZE;
    auto Br_Offset = vThreadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["Slice"]) / width);
    auto Hd_Offset = vThreadIdExpr % (static_cast<uint64_t>(fmhaConfig["Slice"]) / width);
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(Hd_Offset * width + dim2);
    exprs.push_back(Br_Offset);
    return mlir::AffineMap::get(/*dimCount*/3, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "storeTileK") {
    // dims are:[dim0, dim1, dim2]
    // operands are: [threadIdx.x, iv, ivInVector]
    auto threadIdExpr = dim0;
    auto vThreadIdExpr = threadIdExpr + dim1 * BLOCK_SIZE;
    auto Bc_Offset = vThreadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["Slice"]) / width);
    auto Hd_Offset = vThreadIdExpr % (static_cast<uint64_t>(fmhaConfig["Slice"]) / width);
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(Hd_Offset * width + dim2);
    exprs.push_back(Bc_Offset);
    return mlir::AffineMap::get(/*dimCount*/3, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "loadFragQ") {
    // dims are:[dim0, dim1, dim2]
    // operands are: [threadIdx.x, hd_inner, iv]
    auto threadIdExpr = dim0;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]);

    auto ylane_s = laneId.floorDiv(LaneX_S);

    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim1);
    exprs.push_back(warpId * LaneY_S * fmhaConfig["BrTileS"] + dim2 * LaneY_S * width + ylane_s * width);
    return mlir::AffineMap::get(/*dimCount*/3, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());    
  } else if (mapIdentifier == "loadFragK") {
    // dims are:[dim0, dim1, dim2]
    // operands are: [threadIdx.x, hd_inner, iv]
    auto threadIdExpr = dim0;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]);

    auto xlane_s = laneId % (LaneX_S);

    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim1);
    exprs.push_back(dim2 * LaneX_S * width + xlane_s * width);
    return mlir::AffineMap::get(/*dimCount*/3, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());    
  } else if (mapIdentifier == "brIdxS") {
    // dims are:[dim0, dim1]
    // operands are: [threadIdx.x, iv]
    auto threadIdExpr = dim0;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]);

    auto ylane_s = laneId.floorDiv(LaneX_S);
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(warpId * LaneY_S * fmhaConfig["BrTileS"] + dim1.floorDiv(width) * LaneY_S * width + ylane_s * width + dim1 % width);
    return mlir::AffineMap::get(/*dimCount*/2, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
  } else if (mapIdentifier == "storeTileP") {
    // dims are:[dim0, dim1, dim2]
    // operands are: [threadIdx.x, bc, br]
    auto threadIdExpr = dim0;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]);

    auto ylane_s = laneId.floorDiv(LaneX_S);
    auto xlane_s = laneId % (LaneX_S);
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim1.floorDiv(width) * LaneX_S * width + xlane_s * width + dim1 % width);
    exprs.push_back(warpId * LaneY_S * fmhaConfig["BrTileS"] + dim2 * LaneY_S + ylane_s * width);
    return mlir::AffineMap::get(/*dimCount*/3, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
  } else if (mapIdentifier == "readTileS") {
    // dims are:[dim0, dim1]
    // operands are: [bc, br]
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0);
    exprs.push_back(dim1);
    return mlir::AffineMap::get(/*dimCount*/2, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "loadFactor") {
    // dims are:[dim0, dim1]
    // operands are: [threadIdx.x, br]
    auto threadIdExpr = dim0;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]);

    auto xwarp_o = warpId % fmhaConfig["WarpX_O"];
    auto ywarp_o = warpId.floorDiv(fmhaConfig["WarpX_O"]);

    auto ylane_o = laneId.floorDiv(LaneX_O);

    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(ywarp_o * LaneY_O * fmhaConfig["BrTileO"] + dim1 * LaneY_O * width + ylane_o * width);
    return mlir::AffineMap::get(/*dimCount*/2, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "loadTileV") {
    // dims are:[dim0, dim1, dim2, dim3, dim4, dim5]
    // operands are: [blockIdx.z, blockIdx.y, threadIdx.x, k_outer, bc_outer, iv]
    // iv represent a block copy for iv times. 
    auto threadIdExpr = dim2;
    auto vThreadIdExpr = threadIdExpr + dim5 * BLOCK_SIZE;
    auto Bc_Offset = vThreadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["Hd"]) / width);
    auto Hd_Offset = vThreadIdExpr % (static_cast<uint64_t>(fmhaConfig["Hd"]) / width); 
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0);
    exprs.push_back(dim1);
    exprs.push_back(dim3 + Bc_Offset + dim4);
    exprs.push_back(Hd_Offset * width);
    return mlir::AffineMap::get(/*dimCount*/6, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else if (mapIdentifier == "storeTileV") {
    // dims are:[dim0, dim1]
    // operands are: [threadIdx.x, iv]
    auto threadIdExpr = dim0;
    auto vThreadIdExpr = threadIdExpr + dim1 * BLOCK_SIZE;
    auto Bc_Offset = vThreadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["Hd"]) / width);
    auto Hd_Offset = vThreadIdExpr % (static_cast<uint64_t>(fmhaConfig["Hd"]) / width);
    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(Bc_Offset);
    exprs.push_back(Hd_Offset * width);
    return mlir::AffineMap::get(/*dimCount*/2, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());    
  } else if (mapIdentifier == "loadFragP") {
    // dims are:[dim0, dim1, dim2, dim3]
    // operands are: [threadIdx.x, bc_outer, bc_inner, iv]
    auto threadIdExpr = dim0;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]);

    auto ywarp_o = warpId.floorDiv(fmhaConfig["WarpX_O"]);

    auto ylane_o = laneId.floorDiv(LaneX_O);

    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim1 + dim2);
    exprs.push_back(ywarp_o * LaneY_O * fmhaConfig["BrTileO"] + dim3 * width * LaneY_O + ylane_o * width);
    return mlir::AffineMap::get(/*dimCount*/4, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());   
  } else if (mapIdentifier == "loadFragV") {
    // dims are:[dim0, dim1, dim2]
    // operands are: [threadIdx.x, bc_inner, iv]
    auto threadIdExpr = dim0;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]);

    auto xwarp_o = warpId % fmhaConfig["WarpX_O"];
    auto xlane_o = laneId % LaneX_O;

    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim1);
    exprs.push_back(xwarp_o * LaneX_O * fmhaConfig["HdTileO"] + dim2 * width * LaneX_O + xlane_o * width);
    return mlir::AffineMap::get(/*dimCount*/3, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());   
  } else if (mapIdentifier == "brIdxO") {
    // dims are:[dim0, dim1]
    // operands are: [threadIdx.x, iv]
    auto threadIdExpr = dim0;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]);

    auto ywarp_o = warpId.floorDiv(fmhaConfig["WarpX_O"]);

    auto ylane_o = laneId.floorDiv(LaneX_O);

    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(ywarp_o * LaneY_O * fmhaConfig["BrTileO"] + dim1 * LaneY_O * width + ylane_o * width);
    return mlir::AffineMap::get(/*dimCount*/2, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
  } else if (mapIdentifier == "storeTileO") {
    // dims are:[dim0, dim1, dim2, dim3, dim4, dim5]
    // operands are: [blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.x, br, hd]
    auto threadIdExpr = dim0;
    auto warpId = threadIdExpr.floorDiv(static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]));
    auto laneId = threadIdExpr % static_cast<uint64_t>(fmhaConfig["WARP_SIZE"]);

    auto xwarp_o = warpId % fmhaConfig["WarpX_O"];
    auto ywarp_o = warpId.floorDiv(fmhaConfig["WarpX_O"]);

    auto ylane_o = laneId.floorDiv(LaneX_O);
    auto xlane_o = laneId % (LaneX_O);

    auto Br_Offset = ywarp_o * LaneY_O * fmhaConfig["BrTileO"] + dim4.floorDiv(width) * width * LaneY_O + ylane_o * width + dim4 % width;
    auto Hd_Offset = xwarp_o * LaneX_O * fmhaConfig["HdTileO"] + dim5 * LaneX_O + xlane_o * width;

    llvm::SmallVector<mlir::AffineExpr> exprs;
    exprs.push_back(dim0);
    exprs.push_back(dim1);
    exprs.push_back(dim2 * fmhaConfig["Br"] + Br_Offset);
    exprs.push_back(Hd_Offset);
    return mlir::AffineMap::get(/*dimCount*/6, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  } else {
    assert(false);
  }
}

void initMaxSum(mlir::OpBuilder& builder, int64_t lowerBound, int64_t upperBound, int64_t step, mlir::Value tid,
      mlir::Value smemMax, mlir::Value flt_min, mlir::Value smemSum, mlir::Value zero) {
  auto initMaxSumBody = [&](mlir::OpBuilder &kBuilder, mlir::Location kLoc, mlir::Value iv,
                      mlir::ValueRange iterArgs) {
    mlir::OpBuilder::InsertionGuard kGuard(kBuilder);

    auto yieldOp = builder.create<mlir::AffineYieldOp>(builder.getUnknownLoc());
    builder.setInsertionPoint(yieldOp);

    llvm::SmallVector<mlir::AffineExpr> exprs;
    llvm::SmallVector<bool> eqFlags;
    // iv + 2 * step <= ub
    //-> ub - 2 * step - iv >= 0
    auto dim0 = builder.getAffineDimExpr(0);
    auto dim1 = builder.getAffineDimExpr(1);
    exprs.push_back(upperBound - 1 - dim0 - dim1);
    eqFlags.push_back(false);///< true: ==, false: >=
    auto cst = mlir::IntegerSet::get(2, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), llvm::ArrayRef<bool>(eqFlags));
    auto ifOp = builder.create<mlir::AffineIfOp>(builder.getUnknownLoc(), cst, mlir::ValueRange{tid, iv}, 
                                                /*withElseRegion=*/false);

    auto ifBody = ifOp.getBody();
    builder.setInsertionPointToStart(ifBody);

    mlir::AffineMap map = mlir::AffineMap::get(/*dimCount*/2, 0, llvm::ArrayRef<mlir::AffineExpr>({dim0 + dim1}), builder.getContext());

    builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), 
      zero, smemSum, map, mlir::ValueRange{tid, iv});
    builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), 
      flt_min, smemMax, map, mlir::ValueRange{tid, iv});

  };
  auto initMaxSum = builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), 
    lowerBound, upperBound, step, /*iterArgs=llvm::None*/ llvm::None, initMaxSumBody);
}

void FMHAOptimizer::softmaxIR(mlir::OpBuilder& builder, mlir::Value tileS, mlir::Value rowMax, mlir::Value smMax, mlir::Value rowSum, 
  mlir::Value smSum, mlir::Value smFac, mlir::Value zero, mlir::Value flt_min, mlir::Value tid) {
  auto br_loop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BrTileS"], 1);
  auto br = br_loop.getInductionVar();
  builder.setInsertionPointToStart(br_loop.getBody());
  builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), flt_min, rowMax, mlir::ValueRange({br}));
  builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), zero, rowSum, mlir::ValueRange({br}));

  auto bc_loop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BcTileS"], 1);
  auto bc = bc_loop.getInductionVar();
  builder.setInsertionPointToStart(bc_loop.getBody());

  auto tmp1 = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), rowMax, mlir::ValueRange({br}));
  auto tmp2 = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), tileS, mlir::ValueRange({bc, br}));
  auto tmp3 = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(), tmp1, tmp2);

  auto tmp4 = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), rowMax, mlir::ValueRange({br}));
  auto tmp5 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), tmp1.getResult(), tmp3.getResult());
  auto tmp6 = builder.create<mlir::math::ExpOp>(builder.getUnknownLoc(), tmp5.getResult());
  auto tmp7 = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), rowSum, mlir::ValueRange({br}));
  auto tmp8 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), tmp6.getResult(), tmp7.getResult());

  auto tmp9 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), tmp2.getResult(), tmp3.getResult());
  auto tmp10 = builder.create<mlir::math::ExpOp>(builder.getUnknownLoc(), tmp9.getResult());

  auto tmp11 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), tmp8.getResult(), tmp10.getResult());
  builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), tmp11.getResult(), rowMax, mlir::ValueRange({br}));
  
  builder.setInsertionPointAfter(br_loop);
  auto br_shfl_loop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BrTileS"], 1);
  auto br_sf = br_shfl_loop.getInductionVar();
  builder.setInsertionPointToStart(br_shfl_loop.getBody());

  for (int i = 1; i < fmhaConfig["Bc"] / fmhaConfig["BcTileS"]; i *= 2) {
    auto tmp1 = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), rowMax, mlir::ValueRange({br_sf}));
    auto tmp2 = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), rowSum, mlir::ValueRange({br_sf}));

    auto tmp3 = builder.create<mlir::gpu::ShuffleOp>(builder.getUnknownLoc(), tmp1.getResult(), i, 
        fmhaConfig["Bc"] / fmhaConfig["BcTileS"], mlir::gpu::ShuffleMode::DOWN);
    auto tmp4 = builder.create<mlir::gpu::ShuffleOp>(builder.getUnknownLoc(), tmp2.getResult(), i, 
        fmhaConfig["Bc"] / fmhaConfig["BcTileS"], mlir::gpu::ShuffleMode::DOWN);
    
    auto tmp5 = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(), tmp1.getResult(), tmp3.getResult(0));

    auto tmp6 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), tmp1.getResult(), tmp5.getResult());
    auto tmp7 = builder.create<mlir::math::ExpOp>(builder.getUnknownLoc(), tmp6.getResult());
    auto tmp8 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), tmp2.getResult(), tmp7.getResult());

    auto tmp9 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), tmp3.getResult(0), tmp5.getResult());
    auto tmp10 = builder.create<mlir::math::ExpOp>(builder.getUnknownLoc(), tmp9.getResult());
    auto tmp11 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), tmp4.getResult(0), tmp10.getResult());

    auto tmp12 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), tmp8.getResult(), tmp11.getResult());
    builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), tmp12.getResult(), rowSum, mlir::ValueRange({br_sf}));
    builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), tmp5.getResult(), rowMax, mlir::ValueRange({br_sf}));
  }

  builder.setInsertionPointAfter(br_shfl_loop);

  llvm::SmallVector<mlir::AffineExpr> exprs;
  llvm::SmallVector<bool> eqFlags;
  // iv + 2 * step <= ub
  //-> ub - 2 * step - iv >= 0
  auto dim0 = builder.getAffineDimExpr(0);
  exprs.push_back(dim0 % (fmhaConfig["Bc"] / fmhaConfig["BcTileS"]));
  eqFlags.push_back(true);///< true: == 0, false: >= 0
  auto cst = mlir::IntegerSet::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), llvm::ArrayRef<bool>(eqFlags));
  auto ifOp = builder.create<mlir::AffineIfOp>(builder.getUnknownLoc(), cst, mlir::ValueRange{tid}, /*withElseRegion=*/false);
  builder.setInsertionPointToStart(ifOp.getBody());

  auto br_wb_loop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BrTileS"], 1);
  auto br_wb = br_wb_loop.getInductionVar();
  builder.setInsertionPointToStart(br_wb_loop.getBody());

  auto map = getAffineMap("brIdxS", builder);
  {
  auto oldMax = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), smMax, map, mlir::ValueRange{tid, br_wb});
  auto oldSum = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), smSum, map, mlir::ValueRange{tid, br_wb});
  auto tmp1 = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), rowMax, mlir::ValueRange({br_wb}));
  auto newMax = builder.create<mlir::arith::MaxFOp>(builder.getUnknownLoc(), oldMax.getResult(), tmp1.getResult());
  auto tmp3 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), tmp1.getResult(), newMax.getResult());
  auto rowFactor = builder.create<mlir::math::ExpOp>(builder.getUnknownLoc(), tmp3.getResult());
  builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), newMax.getResult(), smMax, map, mlir::ValueRange({tid, br_wb}));
  builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), rowFactor.getResult(), smFac, map, mlir::ValueRange({tid, br_wb}));

  auto tmp4 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), oldSum.getResult(), rowFactor.getResult());

  auto tmp5 = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), rowSum, mlir::ValueRange({br_wb}));
  auto tmp6 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), tmp1.getResult(), newMax.getResult());
  auto tmp7 = builder.create<mlir::math::ExpOp>(builder.getUnknownLoc(), tmp6.getResult());
  auto tmp8 = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), tmp5.getResult(), tmp7.getResult());

  auto tmp9 = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), tmp4.getResult(), tmp8.getResult());

  builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), tmp9.getResult(), smSum, map, mlir::ValueRange({tid, br_wb}));
  builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), newMax.getResult(), rowMax, mlir::ValueRange({br_wb}));
  }
  builder.setInsertionPointAfter(ifOp);
  ///< Write P to shared memory.
  ///< Update rowMax within the same row.
  auto br_broadcast_loop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BrTileS"], 1);
  auto br_broadcast = br_broadcast_loop.getInductionVar();
  builder.setInsertionPointToStart(br_broadcast_loop.getBody());
  {
  auto tmp1 = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), rowMax, mlir::ValueRange({br_broadcast}));
  auto tmp2 = builder.create<mlir::gpu::ShuffleOp>(builder.getUnknownLoc(), tmp1.getResult(), 0, 
      fmhaConfig["Bc"] / fmhaConfig["BcTileS"], mlir::gpu::ShuffleMode::IDX);
  builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), tmp2.getResult(0), rowMax, mlir::ValueRange({br_broadcast}));
  }
  builder.setInsertionPointAfter(br_broadcast_loop);
  
  ///< factor each element by exp.
  ///< tileS[bc][br] = exp(tileS[bc][br] - rowMax[br]);
  auto outerLoop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BcTileS"], 1);
  auto ip = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(outerLoop.getBody());
  auto innerLoop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BrTileS"], 1);
  builder.setInsertionPointToStart(innerLoop.getBody());
  {
    auto bc = outerLoop.getInductionVar();
    auto br = innerLoop.getInductionVar();
    auto ld_s = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), tileS, mlir::ValueRange({bc, br}));
    auto ld_row_max = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), rowMax, mlir::ValueRange({br}));
    auto tmp1 = builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(), ld_s.getResult(), ld_row_max.getResult());
    auto tmp2 = builder.create<mlir::math::ExpOp>(builder.getUnknownLoc(), tmp1.getResult());
    builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), tmp2.getResult(), tileS, mlir::ValueRange({bc, br}));
     
  }
  builder.restoreInsertionPoint(ip);
}

void FMHAOptimizer::applyOptimzer(mlir::ModuleOp& module, mlir::OpBuilder& builder) {
  for (auto& item : call2callsMap) {
    auto call2Matmul = item.first;
    auto call2Softmax = item.second[0];
    auto call2Matmul2 = item.second[1];
    auto matmul = Analyzer::collectFunctions(module, call2Matmul.getCallee().str())[0];
    auto softmax = Analyzer::collectFunctions(module, call2Softmax.getCallee().str())[0];
    auto matmul2 = Analyzer::collectFunctions(module, call2Matmul2.getCallee().str())[0];
    auto buf = call2bufferMap[call2Matmul];
    auto matmul1Desc = buf.matmul1;
    auto Q = buf.Q;
    auto K = buf.K;
    auto V = buf.V;
    auto O = buf.O;

    auto QType = Q.getType().dyn_cast<mlir::MemRefType>();
    auto elementType = QType.getElementType();

    const int seq_len = matmul1Desc.m;
    const int head_dim = matmul1Desc.k;

    ///< New Func Name.
    auto fusedFuncName = std::string({"Fused_Multi_Head_Attention"});
    for (auto b : matmul1Desc.batch) {
      fusedFuncName += "_";
      fusedFuncName += std::to_string(b);
    }
    fusedFuncName += "_SL" + std::to_string(seq_len) + "_HD" + std::to_string(head_dim);
    auto ip = builder.saveInsertionPoint();
    builder.setInsertionPoint(matmul);
    auto funcOp = buildFuction(module, builder, fusedFuncName, {Q.getType(), K.getType(), V.getType(), O.getType()}, {O.getType()});
    builder.setInsertionPointAfter(call2Matmul2);
    auto callOp = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), funcOp, mlir::ValueRange({Q, K, V, O}));
    funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));

    ///< Erase old three function calls.
    call2Matmul2.getResult(0).replaceAllUsesWith(callOp.getResult(0));
    call2Matmul2.erase();
    call2Softmax.erase();
    call2Matmul.erase();

    auto& bodyBlock = funcOp.front();
    auto newArgs = bodyBlock.getArguments();
    builder.setInsertionPointToStart(&bodyBlock);
    auto returnOp = builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), newArgs[3]);

    builder.setInsertionPoint(returnOp);

    ///<===================================  Generate Fused Kernel ==========================
    Q = newArgs[0];
    K = newArgs[1];
    V = newArgs[2];
    O = newArgs[3];
    std::vector<int64_t> bounds;

    for (auto b : matmul1Desc.batch) bounds.push_back(b);

    auto Hd = head_dim;
    auto Br = fmhaConfig["HdxBr"] / Hd;
    auto Bc = fmhaConfig["BrxBc"] / Br;
    auto Slice = fmhaConfig["Slice"];
    auto BLOCK_SIZE = fmhaConfig["BLOCK_SIZE"];

    fmhaConfig["Hd"] = Hd;
    fmhaConfig["Br"] = Br;
    fmhaConfig["Bc"] = Bc;

    bounds.push_back(seq_len / Br);
    bounds.push_back(fmhaConfig["BLOCK_SIZE"]);

    mlir::SmallVector<int64_t, 8> lowerBounds(bounds.size(), /*Value=*/0);
    mlir::SmallVector<int64_t, 8> steps(bounds.size(), /*Value=*/1);
    mlir::SmallVector<int64_t, 8> upperBounds(bounds.begin(), bounds.end());
    mlir::buildAffineLoopNest(
      builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {});
    
    auto loops = Analyzer::collectFuncLoops(funcOp);
    auto batch = loops[0];
    auto head_num = loops[1];
    auto seq_frag = loops[2];
    auto blockLoop = loops[3];

    auto gridLevel = Rewriter::parallel({batch, head_num, seq_frag});
    auto blockLevel = Rewriter::parallel({blockLoop});
    funcOp->setAttr(std::string("func.state"), builder.getStringAttr("gpu"));
    auto smQ = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::shared, {Slice, Br}, elementType);
    auto smK = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::shared, {Slice, Bc}, elementType);
    auto smV = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::shared, {Slice, Bc}, elementType);
    auto smP = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::shared, {Br, Bc}, elementType);
    auto smMax = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::shared, {Br}, elementType);
    auto smSum = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::shared, {Br}, elementType);
    auto smFac = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::shared, {Br}, elementType);

    auto BrTileS = fmhaConfig["BrTileS"];
    auto BcTileS = fmhaConfig["BcTileS"];
    auto BrTileO = fmhaConfig["BrTileO"];
    auto HdTileO = fmhaConfig["HdTileO"];

    auto tileO = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {BrTileO, HdTileO}, elementType);

    builder.setInsertionPointAfter(Analyzer::getLastOp<mlir::memref::AllocOp>(blockLevel));

    auto zero = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), 
        builder.getFloatAttr(elementType, 0));
    auto flt_min = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), 
        builder.getFloatAttr(elementType, -FLT_MAX));

    Rewriter::set_buffer(builder, tileO, zero.getResult());
    
    initMaxSum(builder, 0, Br, BLOCK_SIZE, blockLevel.getIVs()[0], smMax, flt_min.getResult(), smSum, zero.getResult());

    auto outer_reduce = Rewriter::create_constant_loop(builder, 0, seq_len, Bc);
    
    auto ldgQ = Rewriter::alloc_buffer(outer_reduce, MemorySpace::local, {Slice * Br / BLOCK_SIZE}, elementType);
    auto ldgK = Rewriter::alloc_buffer(outer_reduce, MemorySpace::local, {Slice * Bc / BLOCK_SIZE}, elementType);

    auto tileS = Rewriter::alloc_buffer(outer_reduce, MemorySpace::local, {BcTileS, BrTileS}, elementType);

    builder.setInsertionPointAfter(Analyzer::getLastOp<mlir::memref::AllocOp>(outer_reduce));
    Rewriter::set_buffer(builder, tileS, zero.getResult());

    auto hd_outer = Rewriter::create_constant_loop(builder, 0, Hd, Slice);
    auto bar1 = Rewriter::barrier(hd_outer, Position::begin);

    builder.setInsertionPointAfter(bar1);

    Rewriter::read(builder, Q, ldgQ, getAffineMap("loadTileQ", builder), {gridLevel.getIVs()[0], gridLevel.getIVs()[1], 
      gridLevel.getIVs()[2], blockLevel.getIVs()[0], hd_outer.getInductionVar()}, fmhaConfig["Width"]);
    Rewriter::read(builder, K, ldgK, getAffineMap("loadTileK", builder), {gridLevel.getIVs()[0], gridLevel.getIVs()[1], 
      blockLevel.getIVs()[0], outer_reduce.getInductionVar(), hd_outer.getInductionVar()}, fmhaConfig["Width"]);
    Rewriter::write(builder, ldgQ, smQ, getAffineMap("storeTileQ", builder), {blockLevel.getIVs()[0]}, fmhaConfig["Width"]);
    auto writeSmK = Rewriter::write(builder, ldgK, smK, getAffineMap("storeTileK", builder), {blockLevel.getIVs()[0]}, fmhaConfig["Width"]);

    auto bar2 = Rewriter::barrier(writeSmK, Position::after);

    auto fragQ = Rewriter::alloc_buffer(bar2, Position::after, MemorySpace::local, {BrTileS}, elementType);
    auto fragK = Rewriter::alloc_buffer(fragQ.getDefiningOp(), Position::after, MemorySpace::local, {BcTileS}, elementType);

    auto hd_inner = Rewriter::create_constant_loop(builder, 0, Slice, 1);
    builder.setInsertionPointToStart(hd_inner.getBody());
    Rewriter::read(builder, smQ, fragQ, getAffineMap("loadFragQ", builder), {blockLevel.getIVs()[0], hd_inner.getInductionVar()}, 
      fmhaConfig["Width"]);
    Rewriter::read(builder, smK, fragK, getAffineMap("loadFragK", builder), {blockLevel.getIVs()[0], hd_inner.getInductionVar()}, 
      fmhaConfig["Width"]);

    Rewriter::outer_product(builder, tileS, fragK, fragQ, BcTileS, BrTileS);

    builder.setInsertionPointAfter(hd_outer);

    auto rowMax = Rewriter::alloc_buffer(hd_outer, Position::after, MemorySpace::local, {BrTileS}, elementType);
    auto rowSum = Rewriter::alloc_buffer(rowMax.getDefiningOp(), Position::after, MemorySpace::local, {BrTileS}, elementType);

    softmaxIR(builder, tileS, rowMax, smMax, rowSum, smSum, smFac, zero.getResult(), flt_min.getResult(), blockLevel.getIVs()[0]);

    ///< Write P(S) to shared memory.
    auto outerLoop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BcTileS"], 1);
    builder.setInsertionPointToStart(outerLoop.getBody());
    auto innerLoop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BrTileS"], fmhaConfig["Width"]);
    builder.setInsertionPointToStart(innerLoop.getBody());
    {
      auto bc = outerLoop.getInductionVar();
      auto br = innerLoop.getInductionVar();
      auto vectorType = mlir::VectorType::get(fmhaConfig["Width"], tileS.getType().dyn_cast<mlir::MemRefType>().getElementType());
      auto ld = builder.create<mlir::AffineVectorLoadOp>(builder.getUnknownLoc(), vectorType, tileS, getAffineMap("readTileS", builder), mlir::ValueRange({bc, br}));
      auto st = builder.create<mlir::AffineVectorStoreOp>(builder.getUnknownLoc(), ld.getResult(), smP, getAffineMap("storeTileP", builder), mlir::ValueRange{blockLevel.getIVs()[0], bc, br}); 
    }
    auto bar3 = Rewriter::barrier(outerLoop, Position::after);
    auto factor = Rewriter::alloc_buffer(bar3, Position::after, MemorySpace::local, {BrTileO}, elementType);
    builder.setInsertionPointAfter(factor.getDefiningOp());
    Rewriter::read(builder, smFac, factor, getAffineMap("loadFactor", builder), {blockLevel.getIVs()[0]}, fmhaConfig["Width"]);

    ///< Refactor tileO.
    {
    auto outerLoop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BrTileO"], 1);
    builder.setInsertionPointToStart(outerLoop.getBody());
    auto innerLoop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["HdTileO"], 1);
    builder.setInsertionPointToStart(innerLoop.getBody());
  
    auto br = outerLoop.getInductionVar();
    auto hd = innerLoop.getInductionVar();
    
    auto ld_tile_o = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), tileO, mlir::ValueRange({br, hd}));
    auto fac = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), factor, mlir::ValueRange({br}));
    auto tmp = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), ld_tile_o.getResult(), fac.getResult());
    builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), tmp.getResult(), tileO, mlir::ValueRange{br, hd});
    builder.setInsertionPointAfter(outerLoop);
    }

    ///< Compute O = PV
    auto bc_outer = Rewriter::create_constant_loop(builder, 0, Bc, Slice);
    const int LdgV = Slice * Hd / BLOCK_SIZE;
    auto ldgV = Rewriter::alloc_buffer(bc_outer, Position::before, MemorySpace::local, {LdgV}, elementType);

    auto bar4 = Rewriter::barrier(bc_outer, Position::begin);

    builder.setInsertionPointAfter(bar4);

    Rewriter::read(builder, V, ldgV, getAffineMap("loadTileV", builder), {gridLevel.getIVs()[0], gridLevel.getIVs()[1], 
      blockLevel.getIVs()[0], outer_reduce.getInductionVar(), bc_outer.getInductionVar()}, fmhaConfig["Width"]);

    auto writeSmV = Rewriter::write(builder, ldgV, smV, getAffineMap("storeTileV", builder), {blockLevel.getIVs()[0]}, fmhaConfig["Width"]);

    auto bar5 = Rewriter::barrier(writeSmV, Position::after);

    auto fragP = Rewriter::alloc_buffer(bar5, Position::after, MemorySpace::local, {BrTileO}, elementType);
    auto fragV = Rewriter::alloc_buffer(fragP.getDefiningOp(), Position::after, MemorySpace::local, {HdTileO}, elementType);

    auto bc_inner = Rewriter::create_constant_loop(builder, 0, Slice, 1);
    builder.setInsertionPointToStart(bc_inner.getBody());
    Rewriter::read(builder, smP, fragP, getAffineMap("loadFragP", builder), {blockLevel.getIVs()[0], bc_outer.getInductionVar(), bc_inner.getInductionVar()}, 
      fmhaConfig["Width"]);
    Rewriter::read(builder, smV, fragV, getAffineMap("loadFragV", builder), {blockLevel.getIVs()[0], bc_inner.getInductionVar()}, 
      fmhaConfig["Width"]);

    Rewriter::outer_product(builder, tileO, fragP, fragV, BrTileO, HdTileO);

    ///< Load sum
    auto rowSumO = Rewriter::alloc_buffer(outer_reduce, Position::after, MemorySpace::local, {BrTileO}, elementType);
    builder.setInsertionPointAfter(rowSumO.getDefiningOp());
    Rewriter::read(builder, smSum, rowSumO, getAffineMap("brIdxO", builder), {blockLevel.getIVs()[0]}, fmhaConfig["Width"]);
    ///< Refactor tileO
    {
    auto outerLoop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BrTileO"], 1);
    builder.setInsertionPointToStart(outerLoop.getBody());
    auto innerLoop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["HdTileO"], 1);
    builder.setInsertionPointToStart(innerLoop.getBody());
  
    auto br = outerLoop.getInductionVar();
    auto hd = innerLoop.getInductionVar();
    
    auto ld_tile_o = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), tileO, mlir::ValueRange({br, hd}));
    auto tmp = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), rowSumO, mlir::ValueRange({br}));
    auto tmp1 = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), ld_tile_o.getResult(), tmp.getResult());
    builder.create<mlir::AffineStoreOp>(builder.getUnknownLoc(), tmp1.getResult(), tileO, mlir::ValueRange{br, hd});
    builder.setInsertionPointAfter(outerLoop);
    }
    
    ///< Write back O
    {
    auto outerLoop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["BrTileO"], 1);
    builder.setInsertionPointToStart(outerLoop.getBody());
    auto innerLoop = Rewriter::create_constant_loop(builder, 0, fmhaConfig["HdTileO"], fmhaConfig["Width"]);
    builder.setInsertionPointToStart(innerLoop.getBody());
  
    auto br = outerLoop.getInductionVar();
    auto hd = innerLoop.getInductionVar();
    
    auto vectorType = mlir::VectorType::get(fmhaConfig["Width"], tileO.getType().dyn_cast<mlir::MemRefType>().getElementType());
    auto ld = builder.create<mlir::AffineVectorLoadOp>(builder.getUnknownLoc(), vectorType, tileO, mlir::ValueRange({br, hd}));
    auto st = builder.create<mlir::AffineVectorStoreOp>(builder.getUnknownLoc(), ld.getResult(), O, getAffineMap("storeTileO", builder), 
        mlir::ValueRange({gridLevel.getIVs()[0], gridLevel.getIVs()[1], gridLevel.getIVs()[2], blockLevel.getIVs()[0], br, hd})); 
    }
  }
}

}