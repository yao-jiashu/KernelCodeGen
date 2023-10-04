
add_mlir_dialect_library(
  MLIRSchedule
  ScheduleDialect.cc
  ScheduleOps.cc

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/build/include

  DEPENDS
  MLIRScheduleOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
)
