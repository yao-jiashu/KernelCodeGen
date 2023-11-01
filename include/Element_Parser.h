#pragma once
#include "IR/MLIRExtension.h"
#include "ComputeDAG.h"
#include "Expression.h"

namespace KernelCodegen {
using namespace mlir;

class Ele_Parser {
public:

  Ele_Parser() = default;
  int search_var_idx(mlir::Value& variable, std::vector<std::pair<mlir::Value, std::string>>& variables);
  std::string parse_variable_type(mlir::Type a_type, bool with_head, bool as_pointer);
  std::string parse_array_dim(mlir::MemRefType a_type);
  void parse_gpuarth_dim(mlir::AffineForOp forOp, int gpu_arth_dim[][3]);
  std::string parse_for_op_head(mlir::AffineForOp forOp, std::string for_idx_identifier);
  std::string parse_const_op(mlir::Operation* op_pointer, std::string var_identifier);
  std::string parse_muli_op(mlir::Operation* op_pointer, std::string out, std::string in1, std::string in2, bool with_out_type);
  std::string parse_vcopy_op(std::vector<mlir::Operation*> op_pointers, std::vector<std::pair<mlir::Value, std::string>>& variables_dict);
  std::string parse_load_op(mlir::memref::LoadOp& load_op, std::vector<std::pair<mlir::Value, std::string>>& variables_dict);
  std::string parse_store_op(mlir::memref::StoreOp& store_op, std::vector<std::pair<mlir::Value, std::string>>& variables_dict);
  // std::string parse_functhion_argument_type(mlir::BlockArgument blk_arg);
};

}