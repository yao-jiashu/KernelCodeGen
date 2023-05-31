#include "KernelCodegen.h"


namespace KernelCodegen {

void KernelCodegenMachine::autoSchedule() {
  Scheduler s(graph);
  auto loopInfos = s.collectLoops();

  auto m_axis = loopInfos[0].forOp;
  auto n_axis = loopInfos[1].forOp;
  auto k_axis = loopInfos[2].forOp;
  
  auto m_axes = s.split(m_axis, 3, {128, 8});
  auto n_axes = s.split(n_axis, 3, {128, 8});
  auto k_axes = s.split(k_axis, 2, {8});
  
  auto m_outer = m_axes[0];
  auto m_mider = m_axes[1];
  auto m_inner = m_axes[2];
  auto n_outer = n_axes[0];
  auto n_mider = n_axes[1];
  auto n_inner = n_axes[2];
  auto k_outer = k_axes[0];
  auto k_inner = k_axes[1];

  s.reorder({m_outer, n_outer, k_outer, m_mider, n_mider, k_inner, m_inner, n_inner});
}

}