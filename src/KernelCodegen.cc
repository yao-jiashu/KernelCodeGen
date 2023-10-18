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
    graph->dump("bind thread level info");

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

}