#pragma once

enum Log {
  Debug = 0,
  Release = 1,
};
namespace KernelCodeGen {
struct KCGLog {
  static Log level;
};
}