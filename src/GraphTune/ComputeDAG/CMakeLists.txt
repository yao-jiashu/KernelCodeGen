
add_mlir_dialect_library(
  MLIRComputeDAG
  ComputeDAGDialect.cc
  ComputeDAGOps.cc

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/build/include
  # ${PROJECT_SOURCE_DIR}/include/GraphTune/ComputeDAG
  # ${PROJECT_SOURCE_DIR}/build/include/GraphTune/ComputeDAG

  DEPENDS
  MLIRComputeDAGOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
)
