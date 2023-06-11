#include "KernelCodegen.h"


namespace KernelCodegen {

void KernelCodegenMachine::autoSchedule() {
  Scheduler s(graph);
  auto loopInfos = s.collectLoops();

  auto m_axis = loopInfos[0].forOp;
  auto n_axis = loopInfos[1].forOp;
  auto k_axis = loopInfos[2].forOp;

  auto m_axes = s.split(m_axis, 3, {8, 128});
  auto n_axes = s.split(n_axis, 3, {8, 128});

  graph->dumpAndVerify();
  
  
  auto m_outer = m_axes[0];
  auto m_mider = m_axes[1];
  auto m_inner = m_axes[2];
  auto n_outer = n_axes[0];
  auto n_mider = n_axes[1];
  auto n_inner = n_axes[2];


  s.reorder({m_outer, n_outer, m_mider, n_mider, m_inner, n_inner});
  graph->dumpAndVerify();

  s.bind(m_outer, GPUArch::blockIdxY);
  s.bind(n_outer, GPUArch::blockIdxX);
  s.bind(m_mider, GPUArch::threadIdxY);
  s.bind(n_mider, GPUArch::threadIdxX);
  graph->dumpAndVerify();

  auto k_axes = s.split(k_axis, 2, {8});
  graph->dumpAndVerify();

  auto k_outer = k_axes[0];
  auto k_inner = k_axes[1];
  s.reorder({k_outer, k_inner, m_inner, n_inner});
  graph->dumpAndVerify();

  auto insAndOuts = s.collectInputsAndOutputs();
  auto C = insAndOuts[0];
  auto A = insAndOuts[1];
  auto B = insAndOuts[2];

  auto CC = s.cache_write(C, MemorySpace::local, n_mider);
  graph->dumpAndVerify();

  // AA = s.cache_read(A, "shared", [C])
  // BB = s.cache_read(B, "shared", [C])
  // AL = s.cache_read(AA, "local", [C])
  // BL = s.cache_read(BB, "local", [C])
  // CC = s.cache_write(C, "local")

}

}