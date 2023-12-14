#pragma once

#include "IR/IR.h"
#include "Optimizer/Analyzer.h"
#include "enum.h"

#include "mlir/Support/LLVM.h"

#include <vector>

namespace KernelCodeGen {

struct Rewriter {
  Rewriter() = default;

  static mlir::OpBuilder getBuilder(mlir::AffineForOp op, Position pos) {
    switch (pos) {
      case Position::after: {
        mlir::OpBuilder builder(op->getContext());
        builder.setInsertionPointAfter(op);
        return builder;
      }
      case Position::before: {
        mlir::OpBuilder builder(op);
        return builder;
      }
      case Position::begin: {
        return mlir::OpBuilder::atBlockBegin(op.getBody());
      }
      case Position::end: {
        return mlir::OpBuilder::atBlockEnd(op.getBody());
      }
      default:
        assert(false);
    } 
  }

  static std::vector<mlir::Value> getParallelIdx(mlir::AffineParallelOp parallelLevel) {
    auto dim = parallelLevel.getNumDims();
    std::vector<mlir::Value> idxes;
    auto ivs = parallelLevel.getIVs();
    for (auto iv : ivs) {
      idxes.push_back(iv);
    }
    return idxes;
  }

  /// @brief 
  /// @param forOp 
  /// @param num_output 
  /// @param factors 
  /// @return 
  static std::vector<mlir::AffineForOp> split(mlir::AffineForOp forOp, 
                                              uint64_t num_output, std::vector<int64_t>&& factors);
  
  /// @brief 
  /// @param loops 
  /// @return 
  static mlir::Value bufferizeLoopCarryVar(std::vector<mlir::AffineForOp>& loops);

  /// @brief 
  /// @param forOp 
  static void reorder(const std::vector<mlir::AffineForOp>& forOp);

  /// @brief 
  /// @param forOp 
  /// @return 
  static mlir::AffineParallelOp parallel(const std::vector<mlir::AffineForOp>& forOp);

  /// @brief 
  /// @param parallelLevel 
  /// @param ms 
  /// @param shape 
  /// @param dtype 
  /// @return 
  static mlir::Value alloc_buffer(mlir::AffineParallelOp parallelLevel, MemorySpace ms, const std::vector<int64_t> shape, mlir::Type dtype);

  /// @brief 
  /// @param src 
  /// @param dst 
  /// @param map 
  /// @param operands 
  /// @param width 
  /// @param compute_at 
  /// @param pos 
  /// @return 
  static mlir::AffineForOp read(mlir::Value src, mlir::Value dst, mlir::AffineMap map, 
                                   llvm::SmallVector<mlir::Value> operands, int64_t width,
                                   mlir::AffineForOp compute_at, Position pos);

  /// @brief 
  /// @param src 
  /// @param dst 
  /// @param map 
  /// @param operands 
  /// @param width 
  /// @param compute_at 
  /// @param pos 
  /// @return 
  static mlir::AffineForOp write(mlir::Value src, mlir::Value dst, mlir::AffineMap map, 
                                   llvm::SmallVector<mlir::Value> operands, int64_t width,
                                   mlir::AffineForOp compute_at, Position pos);

  /// @brief 
  /// @param compute_at 
  /// @param pos 
  /// @return 
  static mlir::gpu::BarrierOp barrier(mlir::AffineForOp compute_at, Position pos);

  /// @brief 
  /// @param readOrWrite 
  /// @param width 
  /// @return 
  static mlir::AffineForOp vectorize(mlir::AffineForOp readOrWrite, int64_t width);

  /// @brief 
  /// @param scope 
  /// @param src 
  /// @param cached 
  /// @param map 
  /// @param operands 
  static void cache_read(mlir::AffineForOp scope, mlir::Value src, mlir::Value cached, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands);

  /// @brief 
  /// @param scope 
  /// @param src 
  /// @param cached 
  /// @param map 
  /// @param operands 
  static void cache_write(mlir::AffineForOp scope, mlir::Value src, mlir::Value cached, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands);

  /// @brief 
  /// @param parallelLevel 
  /// @param dst 
  /// @return 
  static std::vector<std::vector<mlir::AffineForOp>> get_write(mlir::AffineParallelOp parallelLevel, mlir::Value dst);


  /// @brief double buffer for `buffer`, and pipeline for `readBody`. all of them are computead at `compute_at`
  /// @param readBody 
  /// @param buffer 
  /// @param compute_at 
  /// @return
  static std::vector<std::vector<mlir::AffineForOp>> pipeline(std::vector<mlir::AffineForOp> readBodys, mlir::Value& buffer, mlir::AffineForOp compute_at);

  static void change_double_buffer(mlir::AffineForOp, mlir::Value buffer);

  /// @brief 
  static void detach_last_loop(mlir::AffineForOp forOp);

  /// @brief 
  /// @param srcOp 
  /// @param dstOp 
  /// @param pos 
  static void schedule(mlir::Operation* srcOp, mlir::Operation* dstOp, Position pos);


  /// @brief 
  /// @param srcOp 
  /// @param dstOp 
  /// @param pos 
  static void extract_loop(mlir::Operation* srcOp, mlir::AffineForOp forOp, int64_t iteration);

  /// @brief 
  /// @param op 
  static void take_off_true_if(mlir::ModuleOp module);

  /// @brief 
  /// @param op 
  static void delete_false_if(mlir::ModuleOp module);

  /// @brief 
  /// @param forOp 
  static void unroll(mlir::ModuleOp module, mlir::function_ref<bool(mlir::AffineForOp)> unrollCheckFn);

  /// @brief 
  /// @param forOp 
  static void unrollAttribute(mlir::ModuleOp module, mlir::function_ref<bool(mlir::AffineForOp)> unrollCheckFn);

  /// @brief 
  /// @param module 
  static void loweringAffineDialect(mlir::ModuleOp module);
};

}