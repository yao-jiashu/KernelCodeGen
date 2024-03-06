#pragma once

namespace KernelCodeGen {

enum class Target {
  CUDA = 0,
  ROCm = 1,
};

enum class MemorySpace {
  global = 1,
  shared = 2,
  local = 3,
  constant = 4,
  unallocated = 5,
  inplace = 6,
};

enum class Position {
  before = 0,
  after = 1,
  begin = 2,
  end = 3,
};

enum class Layout {
  rowMajor = 0,
  colMajor = 1,
};

}