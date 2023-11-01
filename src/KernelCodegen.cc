#include "KernelCodegen.h"


namespace KernelCodegen {

void KernelCodegenMachine::autoSchedule(GEMMConfig&& config) {
  Scheduler s(graph);
  
  // for functions with gemm operation, we can apply the schedule template below.
  auto gemmFuncs = s.collectFunctions("gemm");

  for (auto func : gemmFuncs) {

    auto loopInfos = s.collectLoops(func);
    auto m_axis = loopInfos[0].forOp;
    auto n_axis = loopInfos[1].forOp;
    auto k_axis = loopInfos[2].forOp;

    // Tiling.
    auto m_axes = s.split(m_axis, 3, {8, 128});
    auto n_axes = s.split(n_axis, 3, {8, 128});
    graph->dump("split MN loops");

    auto m_outer = m_axes[0];
    auto m_mider = m_axes[1];
    auto m_inner = m_axes[2];
    auto n_outer = n_axes[0];
    auto n_mider = n_axes[1];
    auto n_inner = n_axes[2];
    s.reorder({m_outer, n_outer, m_mider, n_mider, m_inner, n_inner});
    graph->dump("reorder MN loops");

    auto by = s.bind(m_outer, GPUArch::blockIdxY);
    auto bx = s.bind(n_outer, GPUArch::blockIdxX);
    auto ty = s.bind(m_mider, GPUArch::threadIdxY);
    auto tx = s.bind(n_mider, GPUArch::threadIdxX);
    auto threads_hierarchy = std::vector<Value> {by, bx, ty, tx};
    // graph->dump("bind thread level info");

    auto k_axes = s.split(k_axis, 2, {8});
    graph->dump("split K loops");

    auto k_outer = k_axes[0];
    auto k_inner = k_axes[1];
    s.reorder({k_outer, k_inner, m_inner, n_inner});
    graph->dump("reorder K loops");

    auto insAndOuts = s.collectInputsAndOutputs();
    auto C = insAndOuts[0];
    auto A = insAndOuts[1];
    auto B = insAndOuts[2];

    // Load A from global to register
    int regs_num_LDA = config.get_lda_times() * config.vector_load_len[0];
    auto A_reg = s.alloc_buffer(func, MemorySpace::local, {1, regs_num_LDA}, "float32");

    auto start_ARow = Mul(by, config.block_workload[0]);
    auto start_ACol = Mul(k_outer.getInductionVar(), config.block_workload[2]);
    auto tensor_A = Tensor(A, {start_ARow, start_ACol}, {config.block_workload[0], config.block_workload[2]}, config.vector_load_len[0]);
    auto tensor_A_reg = Tensor(A_reg, {Constant(0), Constant(0)}, {1, regs_num_LDA}, config.vector_load_len[0]);
    s.memcpy_async(tensor_A_reg, tensor_A, /*compute_at*/k_outer, threads_hierarchy, ThreadScope::block);
    graph->dump("load A from global to register");


    // Load A from global to register
    int regs_num_LDB = config.get_ldb_times() * config.vector_load_len[1];
    auto B_reg = s.alloc_buffer(func, MemorySpace::local, {1, regs_num_LDB}, "float32");

    auto start_BRow = Mul(k_outer.getInductionVar(), config.block_workload[2]);
    auto start_BCol = Mul(bx, config.block_workload[1]);
    auto tensor_B = Tensor(B, {start_BRow, start_BCol}, {config.block_workload[2], config.block_workload[1]}, config.vector_load_len[1]);
    auto tensor_B_reg = Tensor(B_reg, {Constant(0), Constant(0)}, {1, regs_num_LDB}, config.vector_load_len[1]);
    s.memcpy_async(tensor_B_reg, tensor_B, /*compute_at*/k_outer, threads_hierarchy, ThreadScope::block);
    graph->dump("load B from global to register");

    // auto A_sm = s.alloc_buffer(func, MemorySpace::shared, {8, 128}, "float32");
    // graph->dump("alloc shared memory for A");

    // auto B_sm = s.alloc_buffer(func, MemorySpace::shared, {8, 128}, "float32");
    // graph->dump("alloc shared memory for B");

  }
}

// return whither op is contained in accessed_ops
bool inline is_accessed_op(mlir::Operation& op, std::vector<mlir::Operation*>& accessed_ops)
{
  bool result = false;
  for(int i=0; i<accessed_ops.size(); i++)
  {
    if(&op == accessed_ops[i])
      result = true;
  }
  return result;
}

int inline search_var_idx(mlir::Value& variable, std::vector<std::pair<mlir::Value, std::string>>& variables)
{
  int i = 0;
  for(; i<variables.size(); i++)
    if(variable == variables[i].first)
      break;
  if(i == variables.size())
    i = -1;
  return i;
}

std::string KernelCodegenMachine::codeGen() {
  std::string kernel_code;
  Ele_Collecter cl(graph);
  Ele_Parser pser;
  auto Funcs = cl.collectFunctions();
  // printf("funcs op size:%ld\n", Funcs.size());
  
  std::string indentation_size = "  ";  // 2 blank
  
  for(int i =0; i<Funcs.size(); i++)
  {
    std::stringstream func_ss;
    auto func_i = Funcs[i];
    std::vector<std::pair<mlir::Value, std::string>> func_variables;
    int arraies_idx = 0;
    int gpu_arth[2][3] = {1,1,1,1,1,1}; // [grid_dims[3], block_dims[3]]

    // function head -----------------------------------------------------------------------
    std::string func_name = func_i.getName().str();
    if(func_name.find("kernel") != -1)
      func_ss << "__global__ "; // identifier

    assert(func_i.getNumResults() == 0);
    func_ss << "void ";  // result

    func_ss << func_name; // name

    // arguments
    auto func_args = func_i.getArguments();
    func_ss << "(";
    // push all arguments as string
    for (int j=0; j<func_args.size(); j++)
    {
      auto func_arg = func_args[j];
      std::string arg_type_str = pser.parse_variable_type(func_arg.getType(), false, true);
      func_ss << arg_type_str;  // type
      
      int arg_num = func_arg.getArgNumber();
      std::string arg_name = "Arg_" + std::to_string(arg_num);
      func_ss << arg_name; // name
      
      if(j < func_args.size()-1)
        func_ss << ", ";  // separator

      auto func_arg_idx = search_var_idx(func_arg, func_variables);
      assert(func_arg_idx == -1);
      func_variables.push_back({func_arg, arg_name});
    }
    func_ss << ") \n";

    func_ss << "{\n";
    // function body -----------------------------------------------------------------------

    int numRegion = func_i->getNumRegions();
    if (numRegion != 0) 
    {
      auto regions = Funcs[i]->getRegions();
      for(int j=0; j<numRegion; j++)
      {
        auto& blocks = regions[j].getBlocks();
        std::string func_indentation = "";
        for (auto& block : blocks)
        {
          func_indentation = func_indentation + indentation_size;
          auto& ops = block.getOperations();
          for (mlir::Operation& op : ops) {
            auto opname = op.getName().getIdentifier().data();
            if(llvm::isa<mlir::memref::AllocOp>(op))
            {
              mlir::memref::AllocOp alloc_op = llvm::dyn_cast<mlir::memref::AllocOp>(op);
              mlir::MemRefType memref_type = alloc_op.getType();
              std::string array_type_str = pser.parse_variable_type(memref_type, true, false);
              std::string array_identifer = "Array_" + std::to_string(arraies_idx++);
              if(memref_type.getMemorySpaceAsInt() == 2)
                array_identifer = array_identifer + "_sm";

              auto alloc_op_res = alloc_op.getResult();
              assert(search_var_idx(alloc_op_res, func_variables) == -1);
              func_variables.push_back({alloc_op_res, array_identifer});

              std::string array_dim_str = pser.parse_array_dim(memref_type);
              std::string Op_str = array_type_str + array_identifer + array_dim_str + ";\n";
              
              Op_str = func_indentation + Op_str;
              func_ss << Op_str;
            }
            if(llvm::isa<mlir::func::ReturnOp>(op))
            {
              std::string Op_str = "return;\n";
              Op_str = func_indentation + Op_str;
              func_ss << Op_str;
            }
            if(llvm::isa<mlir::AffineForOp>(op))
            {
              auto for_op_indentation = func_indentation;
              std::stack<mlir::Operation*> op_stack;
              std::vector<mlir::Operation*> accessed_ops;
              op_stack.push(&op);
              int debug_flag = 0;
              do
              {
                mlir::Operation* op_i = op_stack.top();
                // access top
                if(!is_accessed_op(*op_i, accessed_ops))
                {
                  // unaccessed op
                  if(llvm::isa<mlir::AffineForOp>(*op_i))
                  {
                    mlir::AffineForOp op_for = dyn_cast<mlir::AffineForOp>(*op_i);
                    mlir::BlockArgument inductionvar = op_for.getInductionVar();
                    if(op_for->getAttr("gpu.parallel_arch") != nullptr)
                    {
                      pser.parse_gpuarth_dim(op_for, gpu_arth);
                      if(search_var_idx(inductionvar, func_variables) == -1)
                      {
                        auto hint_attr = op_for->getAttr("gpu.parallel_arch");
                        auto hint_Str = hint_attr.dyn_cast<StringAttr>().str();

                        int var_idx = func_variables.size();
                        func_variables.push_back({inductionvar, hint_Str});
                      }
                    }
                    else
                    {
                      // else if(op_for->getAttr("schedule.loop_attr") != nullptr)
                      // TODO: support known inductionvar
                      assert(search_var_idx(inductionvar, func_variables) == -1);
                      
                      int var_idx = func_variables.size();
                      std::string var_name = "var" + std::to_string(var_idx);
                      func_variables.push_back({inductionvar, var_name});

                      std::string for_head_string = pser.parse_for_op_head(op_for, var_name);
                      for_head_string = for_op_indentation + for_head_string + " \n";
                      for_head_string = for_head_string + for_op_indentation + "{\n";
                      func_ss << for_head_string;
                      for_op_indentation = for_op_indentation + "  ";
                    }
                  }
                  if(llvm::isa<mlir::arith::ConstantOp>(*op_i))
                  {
                    auto const_op_result = op_i->getResult(0);
                    assert(search_var_idx(const_op_result, func_variables) == -1);
                    
                    int var_idx = func_variables.size();
                    std::string var_name = "const" + std::to_string(var_idx);
                    func_variables.push_back({const_op_result, var_name});

                    std::string const_op_string = pser.parse_const_op(op_i, var_name);
                    func_ss << for_op_indentation << const_op_string << "\n";
                  }
                  if(llvm::isa<mlir::arith::MulIOp>(*op_i))
                  {
                    mlir::arith::MulIOp op_muli = dyn_cast<mlir::arith::MulIOp>(*op_i);
                    std::string op_muli_string;
                    // result
                    auto muli_op_result = op_i->getResult(0);
                    int result_idx = search_var_idx(muli_op_result, func_variables);
                    std::string var_name;
                    if(result_idx == -1)
                    {
                      int var_idx = func_variables.size();
                      var_name = "var" + std::to_string(var_idx);
                      func_variables.push_back({muli_op_result, var_name});
                      op_muli_string = "int " + var_name + " = ";
                    }
                    else
                    {
                      var_name = func_variables[result_idx].second;
                      op_muli_string = var_name + " = ";
                    }

                    // muli op
                    auto muli_rhs = op_muli.getRhs();
                    auto muli_lhs = op_muli.getLhs();
                    int rhs_idx = search_var_idx(muli_rhs, func_variables);
                    int lhs_idx = search_var_idx(muli_lhs, func_variables);
                    if(rhs_idx!=-1 && lhs_idx!=-1)
                    {
                      std::string lhs_string = func_variables[lhs_idx].second.data();
                      std::string rhs_string = func_variables[rhs_idx].second.data();
                      op_muli_string = op_muli_string + lhs_string + " * " + rhs_string + ";";      
                    }
                    func_ss << for_op_indentation << op_muli_string << "\n";
                  }
                  if(llvm::isa<mlir::arith::MulFOp>(*op_i))
                  {
                    mlir::arith::MulFOp op_muli = dyn_cast<mlir::arith::MulFOp>(*op_i);
                    std::string op_muli_string;
                    // result
                    auto muli_op_result = op_i->getResult(0);
                    int result_idx = search_var_idx(muli_op_result, func_variables);
                    std::string var_name;
                    if(result_idx == -1)
                    {
                      int var_idx = func_variables.size();
                      var_name = "var" + std::to_string(var_idx);
                      func_variables.push_back({muli_op_result, var_name});
                      op_muli_string = "float " + var_name + " = ";
                    }
                    else
                    {
                      var_name = func_variables[result_idx].second;
                      op_muli_string = var_name + " = ";
                    }
                    // muli op
                    auto muli_rhs = op_muli.getRhs();
                    auto muli_lhs = op_muli.getLhs();
                    int rhs_idx = search_var_idx(muli_rhs, func_variables);
                    int lhs_idx = search_var_idx(muli_lhs, func_variables);
                    if(rhs_idx!=-1 && lhs_idx!=-1)
                    {
                      std::string lhs_string = func_variables[lhs_idx].second.data();
                      std::string rhs_string = func_variables[rhs_idx].second.data();
                      op_muli_string = op_muli_string + lhs_string + " * " + rhs_string + ";";      
                    }
                    func_ss << for_op_indentation << op_muli_string << "\n";
                  }
                  if(llvm::isa<mlir::arith::DivUIOp>(*op_i))
                  {
                    mlir::arith::DivUIOp op_divui = dyn_cast<mlir::arith::DivUIOp>(*op_i);
                    std::string op_divui_string;
                    // result
                    auto divui_op_result = op_i->getResult(0);
                    int result_idx = search_var_idx(divui_op_result, func_variables);
                    std::string var_name;
                    if(result_idx == -1)
                    {
                      int var_idx = func_variables.size();
                      var_name = "var" + std::to_string(var_idx);
                      func_variables.push_back({divui_op_result, var_name});
                      op_divui_string = "float " + var_name + " = ";
                    }
                    else
                    {
                      var_name = func_variables[result_idx].second;
                      op_divui_string = var_name + " = ";
                    }

                    // divui op
                    auto divui_rhs = op_divui.getRhs();
                    auto divui_lhs = op_divui.getLhs();
                    int rhs_idx = search_var_idx(divui_rhs, func_variables);
                    int lhs_idx = search_var_idx(divui_lhs, func_variables);
                    if(rhs_idx!=-1 && lhs_idx!=-1)
                    {
                      std::string lhs_string = func_variables[lhs_idx].second.data();
                      std::string rhs_string = func_variables[rhs_idx].second.data();
                      op_divui_string = op_divui_string + lhs_string + " / " + rhs_string + ";";
                    }
                    func_ss << for_op_indentation << op_divui_string << "\n";
                  }
                  if(llvm::isa<mlir::arith::RemUIOp>(*op_i))
                  {
                    mlir::arith::RemUIOp op_divui = dyn_cast<mlir::arith::RemUIOp>(*op_i);
                    std::string op_divui_string;
                    // result
                    auto divui_op_result = op_i->getResult(0);
                    int result_idx = search_var_idx(divui_op_result, func_variables);
                    std::string var_name;
                    if(result_idx == -1)
                    {
                      int var_idx = func_variables.size();
                      var_name = "var" + std::to_string(var_idx);
                      func_variables.push_back({divui_op_result, var_name});
                      op_divui_string = "float " + var_name + " = ";
                    }
                    else
                    {
                      var_name = func_variables[result_idx].second;
                      op_divui_string = var_name + " = ";
                    }

                    // divui op
                    auto divui_rhs = op_divui.getRhs();
                    auto divui_lhs = op_divui.getLhs();
                    int rhs_idx = search_var_idx(divui_rhs, func_variables);
                    int lhs_idx = search_var_idx(divui_lhs, func_variables);
                    if(rhs_idx!=-1 && lhs_idx!=-1)
                    {
                      std::string lhs_string = func_variables[lhs_idx].second.data();
                      std::string rhs_string = func_variables[rhs_idx].second.data();
                      op_divui_string = op_divui_string + lhs_string + " % " + rhs_string + ";";
                    }
                    func_ss << for_op_indentation << op_divui_string << "\n";
                  }
                  if(llvm::isa<mlir::arith::AddIOp>(*op_i))
                  {
                    mlir::arith::AddIOp op_muli = dyn_cast<mlir::arith::AddIOp>(*op_i);
                    std::string op_muli_string;
                    // result
                    auto muli_op_result = op_i->getResult(0);
                    int result_idx = search_var_idx(muli_op_result, func_variables);
                    std::string var_name;
                    if(result_idx == -1)
                    {
                      int var_idx = func_variables.size();
                      var_name = "var" + std::to_string(var_idx);
                      func_variables.push_back({muli_op_result, var_name});
                      op_muli_string = "int " + var_name + " = ";        
                    }
                    else
                    {
                      var_name = func_variables[result_idx].second;
                      op_muli_string = var_name + " = ";
                    }

                    // muli op
                    auto muli_rhs = op_muli.getRhs();
                    auto muli_lhs = op_muli.getLhs();
                    int rhs_idx = search_var_idx(muli_rhs, func_variables);
                    int lhs_idx = search_var_idx(muli_lhs, func_variables);
                    if(rhs_idx!=-1 && lhs_idx!=-1)
                    {
                      std::string lhs_string = func_variables[lhs_idx].second.data();
                      std::string rhs_string = func_variables[rhs_idx].second.data();
                      op_muli_string = op_muli_string + lhs_string + " + " + rhs_string + ";";      
                    }
                    func_ss << for_op_indentation << op_muli_string << "\n";
                  }
                  if(llvm::isa<mlir::arith::AddFOp>(*op_i))
                  {
                    mlir::arith::AddFOp op_muli = dyn_cast<mlir::arith::AddFOp>(*op_i);
                    std::string op_muli_string;
                    // result
                    auto muli_op_result = op_i->getResult(0);
                    int result_idx = search_var_idx(muli_op_result, func_variables);
                    std::string var_name;
                    if(result_idx == -1)
                    {
                      int var_idx = func_variables.size();
                      var_name = "var" + std::to_string(var_idx);
                      func_variables.push_back({muli_op_result, var_name});
                      op_muli_string = "float " + var_name + " = ";
                    }
                    else
                    {
                      var_name = func_variables[result_idx].second;
                      op_muli_string = var_name + " = ";
                    }
                    // muli op
                    auto muli_rhs = op_muli.getRhs();
                    auto muli_lhs = op_muli.getLhs();
                    int rhs_idx = search_var_idx(muli_rhs, func_variables);
                    int lhs_idx = search_var_idx(muli_lhs, func_variables);
                    if(rhs_idx!=-1 && lhs_idx!=-1)
                    {
                      std::string lhs_string = func_variables[lhs_idx].second.data();
                      std::string rhs_string = func_variables[rhs_idx].second.data();
                      op_muli_string = op_muli_string + lhs_string + " + " + rhs_string + ";";      
                    }
                    func_ss << for_op_indentation << op_muli_string << "\n";
                  }
                  if(llvm::isa<mlir::memref::LoadOp>(*op_i))
                  {
                    mlir::memref::LoadOp op_load = dyn_cast<mlir::memref::LoadOp>(*op_i);
                    std::string op_load_string = pser.parse_load_op(op_load, func_variables);
                    func_ss << for_op_indentation << op_load_string << "\n";
                  }
                  if(llvm::isa<mlir::memref::StoreOp>(*op_i))
                  {
                    mlir::memref::StoreOp op_store = dyn_cast<mlir::memref::StoreOp>(*op_i);
                    std::string op_store_string = pser.parse_store_op(op_store, func_variables);
                    func_ss << for_op_indentation << op_store_string << "\n";
                  }
                  if(llvm::isa<mlir::memref::CopyOp>(*op_i))
                  {

                  }
                  if(llvm::isa<mlir::memref::SubViewOp>(*op_i))
                  {
                    // printf("--------------------------------\n");
                    auto next_op = op_i->getNextNode();
                    auto nnext_op = next_op->getNextNode();
                    bool copy_pattern = llvm::isa<mlir::memref::SubViewOp>(*next_op);
                    copy_pattern = copy_pattern && llvm::isa<mlir::memref::CopyOp>(*nnext_op);
                    if(copy_pattern)
                    {
                      auto vcopystring = pser.parse_vcopy_op({op_i, next_op, nnext_op}, func_variables);
                      accessed_ops.push_back(next_op);
                      accessed_ops.push_back(nnext_op);
                      func_ss << for_op_indentation << vcopystring << "\n";
                    }
                  }
                  accessed_ops.push_back(op_i);
                }

                // push an op into stack
                bool pop_top = false;
                int num_regions = (*op_i).getNumRegions();
                if(num_regions > 0)
                {
                  assert(num_regions == 1); // only support op that have 1 region
                  auto op_regions = (*op_i).getRegions();
                  for (auto& op_region : op_regions)
                  {
                    auto& op_blocks = op_region.getBlocks();
                    int op_blks_size = op_blocks.size();
                    for (auto& op_block : op_blocks)
                    {
                      auto& op_ops = op_block.getOperations();
                      int op_ops_size = op_ops.size();
                      for (auto& op_op : op_ops) {
                        if(!is_accessed_op(op_op, accessed_ops))
                        {
                          op_stack.push(&op_op);
                          break;  // break op_ops ergodic when meeting an unaccessed op, break to next loop to access it
                        }
                        op_ops_size --;
                      }
                      if(op_ops_size == 0)
                      {
                        op_blks_size --;
                        if(op_blks_size == 0)
                          pop_top = true;
                      }
                      else
                        break;  // break op_blocks ergodic
                    }
                  }
                }
                else
                  pop_top = true;

                // pop top op, when all it's children op and itself are accessed.
                if(pop_top)
                {
                  if(llvm::isa<mlir::AffineForOp>(*op_i))
                  {
                    mlir::AffineForOp op_for = dyn_cast<AffineForOp>(*op_i);
                    if(op_for->getAttr("gpu.parallel_arch") == nullptr)
                    {
                      for_op_indentation = for_op_indentation.substr(0, for_op_indentation.size()-2);
                      func_ss << for_op_indentation + "}\n";
                    }
                  }
                  op_stack.pop();
                }
                debug_flag++;
              } while (!op_stack.empty());
              printf("debug_flag:%d\n", debug_flag);
            }
          }
        }
      }
    }

    func_ss << "}\n";
    std::string func_arg_str = func_ss.str();
    printf("func str: \n%s\n", func_arg_str.data());
    printf("griddim:(%d, %d, %d), blockdim:(%d, %d, %d)\n", 
      gpu_arth[0][0], gpu_arth[0][1], gpu_arth[0][2], 
      gpu_arth[1][0], gpu_arth[1][1], gpu_arth[1][2]);
  }
  return kernel_code;
}

}