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

    auto m_axes = s.split(m_axis, 3, {8, 128});
    auto n_axes = s.split(n_axis, 3, {8, 128});

    graph->dump();
    
    
    auto m_outer = m_axes[0];
    auto m_mider = m_axes[1];
    auto m_inner = m_axes[2];
    auto n_outer = n_axes[0];
    auto n_mider = n_axes[1];
    auto n_inner = n_axes[2];


    s.reorder({m_outer, n_outer, m_mider, n_mider, m_inner, n_inner});
    graph->dump();

    s.bind(m_outer, GPUArch::blockIdxY);
    s.bind(n_outer, GPUArch::blockIdxX);
    s.bind(m_mider, GPUArch::threadIdxY);
    s.bind(n_mider, GPUArch::threadIdxX);
    graph->dump();

    auto k_axes = s.split(k_axis, 2, {8});
    graph->dump();

    auto k_outer = k_axes[0];
    auto k_inner = k_axes[1];
    s.reorder({k_outer, k_inner, m_inner, n_inner});
    graph->dump();

    auto insAndOuts = s.collectInputsAndOutputs();
    auto C = insAndOuts[0];
    auto A = insAndOuts[1];
    auto B = insAndOuts[2];

    auto A_sm = s.alloc_buffer(func, MemorySpace::shared, {8, 128}, "float32");
    auto B_sm = s.alloc_buffer(func, MemorySpace::shared, {8, 128}, "float32");

    auto A_reg = s.alloc_buffer(func, MemorySpace::local, {4}, "float32");
    auto B_reg = s.alloc_buffer(func, MemorySpace::local, {4}, "float32");

    graph->dump();

  }
}

}