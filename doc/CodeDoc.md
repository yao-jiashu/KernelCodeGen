
踏踏实实地干，把自己的想法一点点地实现，然后一点点的丰富

getBlock()是获取父节点的block
getBody()才是获得本节点的block
erase()会删除所有子节点，如果要删除前替换使用loop的迭代变量，需要反序，从内往外删

想clone一个op：使用->clone()而不是.clone()
```C++
    mlir::OpBuilder b(outer->getBlock(), mlir::Block::iterator(outer));
    mlir::BlockAndValueMapping mapper;
    b.clone(*outer, mapper);
```

Operation
Op
OpState
的关系
```c++
//为什么这些执行时失败了，出现段错误
    // while (count < number) {
    //   if (count != 0 && count < number - position) {
    //     iter->erase();
    //   }
    //   count += 1;
    //   --iter;
    // }
    // (--(--iter))->erase();
    
    // (--(--ops.end()))->erase();
    // for (auto iter = ops.rbegin(); iter != ops.rend(); iter++) {
    //   if (count != 0 && count <= 1) iter->erase();
    //   count += 1;
    // }
    // llvm::errs() << "张昌\n";
    // int count = 0;
    // for (auto& op : ops) {
    //   if (count == position) if (&op == inner) llvm::errs() << "滑稽\n";//op.erase();
    //   // if (count >= position && count < opNumber-1) {
    //   //   op.erase();
    //   // }
    //   count += 1;
    // }
```

有时候赋值出错，尝试赋引用