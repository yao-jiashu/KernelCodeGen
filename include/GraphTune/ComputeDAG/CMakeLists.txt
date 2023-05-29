add_mlir_dialect(ComputeDAGOps compute_dag)

# set(LLVM_TARGET_DEFINITIONS ComputeDAGOps.td)
# mlir_tablegen(ComputeDAGOps.h.inc -gen-op-decls)
# mlir_tablegen(ComputeDAGOps.cpp.inc -gen-op-defs)
# mlir_tablegen(ComputeDAGOpsDialect.h.inc -gen-dialect-decls)
# mlir_tablegen(ComputeDAGOpsDialect.cpp.inc -gen-dialect-defs)
# add_public_tablegen_target(MLIRComputeDAGOpsIncGen)

# add_mlir_doc(ComputeDAGOps ComputeDAGOps Dialects/ -gen-op-doc)
# add_mlir_doc(ComputeDAGOps ComputeDAGDialect Dialects/ -gen-dialect-doc)
