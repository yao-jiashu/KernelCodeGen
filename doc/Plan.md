# 2023.11.21

### prority 0
**前端**
- Transformer中的算子实现(描述)
- 方案1：在compute_dag中定义所有的算子
        算子转affine
- 方案2：直接由onnx转affine

**优化器**
- 在affine这一层，描述所有的优化：重点是使用AffineMap来构建所有的映射关系
- [AffineMap](https://mlir.llvm.org/docs/Dialects/Affine/#polyhedral-structures)
- AffineMap使用与分析：Pass重写
- 技术路线：split:(i)->(i, j): A[i], A[map(i, j)]
- tile=split+reorder
- 总结：split, reorder, memcpy(AffineMap), pipeline, vectorize, unroll

**后端**
- 转成 C-Like
- 参考triton


### priority 1

**前端DSL**

**优化器**

**后端**
- 参考triton

具体优化全局决策，Pass只负责执行

总结：
[局部重写]
- split: AffineMap改造，不需要额外的arith操作来计算索引，也不需要affine.apply来取代原有引用，直接在调用处(memref引用处)修改affine映射，便于分析
- reorder: 基于AffineForOp的两两交换来实现，for循环的两两交换要保证正确性，即保证非完美循环嵌套时语句的执行次数的正确性
- parallel: 将完美嵌套的AffineForOp转成AffineParallelOp，根据parallel的层次，可以知道内存的开辟位置。
- memcpy: AffineMap改造，根据对寄存器数组的读合写，分为两种情况：
- - read: 从别的地方读来，写寄存器
- - write: 从寄存器中读，写到别的地方
- - vectorize: 向量化，将affine.load和affine.store转成affine.vector_load和affine.vector_store
- cache_read: 将指定范围内对A的访问，用fragA替代
- cache_write: 将指定范围内对C的修改，用tileC替代
- vectorize: 向量化，将affine.load和affine.store转成affine.vector_load和affine.vector_store
- pipeline: 对buffer进行double，并在指定的循环中，根据循环变量选择操作的buffer
- detach_last_loop:将最后一次循环分离出去
- schedule: 对operation的位置进行调整
- extract_loop: 将循环中第i次迭代的一个operation提取到循环外面
[PASS]
- bufferizeLoopCarryVar[Pass]: 将循环携带变量变为buffer, 即寄存器数组
- take_off_true_if[Pass]: 将条件恒成立的if判断摘掉
- delete_false_if[Pass]: 如果if判断恒不成立则删除if判断，包括if里面的operation
- unroll[Pass]: 对循环次数已知且小于指定次数的循环进行展开

# 2023.12.14

### prority 0
**前端**
- Transformer中的算子的计算描述：TODO(xbk)
- Transformer中常见算子的输入大小：TODO(?)

**优化器**
- Analyzer && Rewrite: Done
- - split, reorder, parallel, read, write, vectorize, cache_read, cache_write, pipeline......
- Matmul Optimizer: Done
- Attention Optimizer: TODO (yjs)
- KernelFusion Optimizer: TODO (yjs)

**后端**
- 生成 Matmul CUDA C 代码: Done(yjs)
- - 正确性测试：Done
- - 性能测试：4096以上的方阵，性能持平 Cublas

# 2023.12.18

### prority 0
**前端**
- Transformer中的算子的计算描述：TODO(xbk)
- - 参考Rewriter实现
- Transformer中常见算子的输入大小：TODO(?)
- - 调研(yjs)

**优化器**
- Analyzer && Rewrite: Done
- - split, reorder, parallel, read, write, vectorize, cache_read, cache_write, pipeline......
- Matmul Optimizer: Done
- Attention Optimizer: TODO (yjs)
- KernelFusion Optimizer: TODO (yjs)

**后端**
- 生成 Matmul CUDA C 代码: Done(yjs)
- - 正确性测试：Done
- - 性能测试：4096以上的方阵，性能持平 Cublas


# 1.15
## 3月前
- flash-attention2 CUDA Core优化实现：1.20
- flash-attention2 参数调优(不需要实现，但需要有逻辑)：1.27
- flash-attention2 MLIR实现：2.29

- batched GEMM MLIR实现： 2.29

- Transformer模型端到端测试：2.29

## 4月前
- batched GEMM（Transformer中参数调优）：3.8
- DCU后端代码生成：3.8
- 算子融合策略 MLIR实现：3.15
- 端到端测试：3.31