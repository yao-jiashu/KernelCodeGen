# Structure
-include
 - MLIR.h header files in llvm-project/mlir
 - MLIREnhance.h  including Dialects which built by our project

# Performance
Description to Matmul
```c++
module @fuse_matmul_relu {
  %0 = memref.alloc() : memref<4096x1024xf32, 1>
  %1 = memref.alloc() : memref<1024x2048xf32, 1>
  %2 = memref.alloc() : memref<4096x2048xf32, 1>
  affine.for %arg0 = 0 to 4096 {
    affine.for %arg1 = 0 to 2048 {
      %cst = arith.constant 0.000000e+00 : f32
      %3 = affine.for %arg2 = 0 to 1024 iter_args(%arg3 = %cst) -> (f32) {
        %4 = affine.load %0[%arg0, %arg2] : memref<4096x1024xf32, 1>
        %5 = affine.load %1[%arg2, %arg1] : memref<1024x2048xf32, 1>
        %6 = arith.mulf %4, %5 : f32
        %7 = arith.addf %6, %arg3 : f32
        affine.yield %7 : f32
      }
      affine.store %3, %2[%arg0, %arg1] : memref<4096x2048xf32, 1>
    }
  }
}
```
split
```c++
module @fuse_matmul_relu {
  %0 = memref.alloc() : memref<4096x1024xf32, 1>
  %1 = memref.alloc() : memref<1024x2048xf32, 1>
  %2 = memref.alloc() : memref<4096x2048xf32, 1>
  affine.for %arg0 = 0 to 4096 step 128 {
    affine.for %arg1 = 0 to 128 step 8 {
      affine.for %arg2 = 0 to 8 {
        affine.for %arg3 = 0 to 2048 step 128 {
          affine.for %arg4 = 0 to 128 step 8 {
            affine.for %arg5 = 0 to 8 {
              %cst = arith.constant 0.000000e+00 : f32
              %3 = affine.for %arg6 = 0 to 1024 iter_args(%arg7 = %cst) -> (f32) {
                %4 = affine.load %0[%arg0 + %arg1 + %arg2, %arg6] : memref<4096x1024xf32, 1>
                %5 = affine.load %1[%arg6, %arg3 + %arg4 + %arg5] : memref<1024x2048xf32, 1>
                %6 = arith.mulf %4, %5 : f32
                %7 = arith.addf %6, %arg7 : f32
                affine.yield %7 : f32
              }
              affine.store %3, %2[%arg0 + %arg1 + %arg2, %arg3 + %arg4 + %arg5] : memref<4096x2048xf32, 1>
            }
          }
        }
      }
    }
  }
}
```
reorder
```c++
module @fuse_matmul_relu {
  %0 = memref.alloc() : memref<4096x1024xf32, 1>
  %1 = memref.alloc() : memref<1024x2048xf32, 1>
  %2 = memref.alloc() : memref<4096x2048xf32, 1>
  affine.for %arg0 = 0 to 4096 step 128 {
    affine.for %arg1 = 0 to 2048 step 128 {
      affine.for %arg2 = 0 to 128 step 8 {
        affine.for %arg3 = 0 to 128 step 8 {
          affine.for %arg4 = 0 to 8 {
            affine.for %arg5 = 0 to 8 {
              %cst = arith.constant 0.000000e+00 : f32
              %3 = affine.for %arg6 = 0 to 1024 iter_args(%arg7 = %cst) -> (f32) {
                %4 = affine.load %0[%arg0 + %arg2 + %arg4, %arg6] : memref<4096x1024xf32, 1>
                %5 = affine.load %1[%arg6, %arg1 + %arg3 + %arg5] : memref<1024x2048xf32, 1>
                %6 = arith.mulf %4, %5 : f32
                %7 = arith.addf %6, %arg7 : f32
                affine.yield %7 : f32
              }
              affine.store %3, %2[%arg0 + %arg2 + %arg4, %arg1 + %arg3 + %arg5] : memref<4096x2048xf32, 1>
            }
          }
        }
      }
    }
  }
}
```
parallel
```c++
module @fuse_matmul_relu {
  %0 = memref.alloc() : memref<4096x1024xf32, 1>
  %1 = memref.alloc() : memref<1024x2048xf32, 1>
  %2 = memref.alloc() : memref<4096x2048xf32, 1>
  affine.parallel (%arg0, %arg1) = (0, 0) to (32, 16) {
    %3 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
    %4 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg1)
    affine.parallel (%arg2, %arg3) = (0, 0) to (16, 16) {
      %5 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg2)
      %6 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg3)
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %cst = arith.constant 0.000000e+00 : f32
          %7 = affine.for %arg6 = 0 to 1024 iter_args(%arg7 = %cst) -> (f32) {
            %8 = affine.load %0[%3 + %5 + %arg4, %arg6] : memref<4096x1024xf32, 1>
            %9 = affine.load %1[%arg6, %4 + %6 + %arg5] : memref<1024x2048xf32, 1>
            %10 = arith.mulf %8, %9 : f32
            %11 = arith.addf %10, %arg7 : f32
            affine.yield %11 : f32
          }
          affine.store %7, %2[%3 + %5 + %arg4, %4 + %6 + %arg5] : memref<4096x2048xf32, 1>
        }
      }
    }
  }
}
```
bufferize loop carry var
```c++
module @fuse_matmul_relu {
  %0 = memref.alloc() : memref<4096x1024xf32, 1>
  %1 = memref.alloc() : memref<1024x2048xf32, 1>
  %2 = memref.alloc() : memref<4096x2048xf32, 1>
  affine.parallel (%arg0, %arg1) = (0, 0) to (32, 16) {
    %3 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
    %4 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg1)
    affine.parallel (%arg2, %arg3) = (0, 0) to (16, 16) {
      %5 = memref.alloc() : memref<8x8xf32, 3>
      %6 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg2)
      %7 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg3)
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %cst = arith.constant 0.000000e+00 : f32
          affine.store %cst, %5[%arg4, %arg5] : memref<8x8xf32, 3>
          affine.for %arg6 = 0 to 1024 {
            %9 = affine.load %5[%arg4, %arg5] : memref<8x8xf32, 3>
            %10 = affine.load %0[%3 + %6 + %arg4, %arg6] : memref<4096x1024xf32, 1>
            %11 = affine.load %1[%arg6, %4 + %7 + %arg5] : memref<1024x2048xf32, 1>
            %12 = arith.mulf %10, %11 : f32
            %13 = arith.addf %12, %9 : f32
            affine.store %13, %5[%arg4, %arg5] : memref<8x8xf32, 3>
          }
          %8 = affine.load %5[%arg4, %arg5] : memref<8x8xf32, 3>
          affine.store %8, %2[%3 + %6 + %arg4, %4 + %7 + %arg5] : memref<4096x2048xf32, 1>
        }
      }
    }
  }
}
```
reorder
```c++
module @fuse_matmul_relu {
  %0 = memref.alloc() : memref<4096x1024xf32, 1>
  %1 = memref.alloc() : memref<1024x2048xf32, 1>
  %2 = memref.alloc() : memref<4096x2048xf32, 1>
  affine.parallel (%arg0, %arg1) = (0, 0) to (32, 16) {
    %3 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
    %4 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg1)
    affine.parallel (%arg2, %arg3) = (0, 0) to (16, 16) {
      %5 = memref.alloc() : memref<8x8xf32, 3>
      %6 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg2)
      %7 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg3)
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %cst = arith.constant 0.000000e+00 : f32
          affine.store %cst, %5[%arg4, %arg5] : memref<8x8xf32, 3>
        }
      }
      affine.for %arg4 = 0 to 1024 {
        affine.for %arg5 = 0 to 8 {
          affine.for %arg6 = 0 to 8 {
            %8 = affine.load %5[%arg5, %arg6] : memref<8x8xf32, 3>
            %9 = affine.load %0[%3 + %6 + %arg5, %arg4] : memref<4096x1024xf32, 1>
            %10 = affine.load %1[%arg4, %4 + %7 + %arg6] : memref<1024x2048xf32, 1>
            %11 = arith.mulf %9, %10 : f32
            %12 = arith.addf %11, %8 : f32
            affine.store %12, %5[%arg5, %arg6] : memref<8x8xf32, 3>
          }
        }
      }
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %8 = affine.load %5[%arg4, %arg5] : memref<8x8xf32, 3>
          affine.store %8, %2[%3 + %6 + %arg4, %4 + %7 + %arg5] : memref<4096x2048xf32, 1>
        }
      }
    }
  }
}
```
split
```c++
module @fuse_matmul_relu {
  %0 = memref.alloc() : memref<4096x1024xf32, 1>
  %1 = memref.alloc() : memref<1024x2048xf32, 1>
  %2 = memref.alloc() : memref<4096x2048xf32, 1>
  affine.parallel (%arg0, %arg1) = (0, 0) to (32, 16) {
    %3 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
    %4 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg1)
    affine.parallel (%arg2, %arg3) = (0, 0) to (16, 16) {
      %5 = memref.alloc() : memref<8x8xf32, 3>
      %6 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg2)
      %7 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg3)
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %cst = arith.constant 0.000000e+00 : f32
          affine.store %cst, %5[%arg4, %arg5] : memref<8x8xf32, 3>
        }
      }
      affine.for %arg4 = 0 to 1024 step 8 {
        affine.for %arg5 = 0 to 8 {
          affine.for %arg6 = 0 to 8 {
            affine.for %arg7 = 0 to 8 {
              %8 = affine.load %5[%arg6, %arg7] : memref<8x8xf32, 3>
              %9 = affine.load %0[%3 + %6 + %arg6, %arg4 + %arg5] : memref<4096x1024xf32, 1>
              %10 = affine.load %1[%arg4 + %arg5, %4 + %7 + %arg7] : memref<1024x2048xf32, 1>
              %11 = arith.mulf %9, %10 : f32
              %12 = arith.addf %11, %8 : f32
              affine.store %12, %5[%arg6, %arg7] : memref<8x8xf32, 3>
            }
          }
        }
      }
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %8 = affine.load %5[%arg4, %arg5] : memref<8x8xf32, 3>
          affine.store %8, %2[%3 + %6 + %arg4, %4 + %7 + %arg5] : memref<4096x2048xf32, 1>
        }
      }
    }
  }
}
```
alloc buffer
```c++
module @fuse_matmul_relu {
  %0 = memref.alloc() : memref<4096x1024xf32, 1>
  %1 = memref.alloc() : memref<1024x2048xf32, 1>
  %2 = memref.alloc() : memref<4096x2048xf32, 1>
  affine.parallel (%arg0, %arg1) = (0, 0) to (32, 16) {
    %3 = memref.alloc() : memref<8x128xf32, 2>
    %4 = memref.alloc() : memref<8x128xf32, 2>
    %5 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
    %6 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg1)
    affine.parallel (%arg2, %arg3) = (0, 0) to (16, 16) {
      %7 = memref.alloc() : memref<4xf32, 3>
      %8 = memref.alloc() : memref<4xf32, 3>
      %9 = memref.alloc() : memref<8xf32, 3>
      %10 = memref.alloc() : memref<8xf32, 3>
      %11 = memref.alloc() : memref<8x8xf32, 3>
      %12 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg2)
      %13 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg3)
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %cst = arith.constant 0.000000e+00 : f32
          affine.store %cst, %11[%arg4, %arg5] : memref<8x8xf32, 3>
        }
      }
      affine.for %arg4 = 0 to 1024 step 8 {
        affine.for %arg5 = 0 to 8 {
          affine.for %arg6 = 0 to 8 {
            affine.for %arg7 = 0 to 8 {
              %14 = affine.load %11[%arg6, %arg7] : memref<8x8xf32, 3>
              %15 = affine.load %0[%5 + %12 + %arg6, %arg4 + %arg5] : memref<4096x1024xf32, 1>
              %16 = affine.load %1[%arg4 + %arg5, %6 + %13 + %arg7] : memref<1024x2048xf32, 1>
              %17 = arith.mulf %15, %16 : f32
              %18 = arith.addf %17, %14 : f32
              affine.store %18, %11[%arg6, %arg7] : memref<8x8xf32, 3>
            }
          }
        }
      }
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %14 = affine.load %11[%arg4, %arg5] : memref<8x8xf32, 3>
          affine.store %14, %2[%5 + %12 + %arg4, %6 + %13 + %arg5] : memref<4096x2048xf32, 1>
        }
      }
    }
  }
}
```
read
```c++
module @fuse_matmul_relu {
  %0 = memref.alloc() : memref<4096x1024xf32, 1>
  %1 = memref.alloc() : memref<1024x2048xf32, 1>
  %2 = memref.alloc() : memref<4096x2048xf32, 1>
  affine.parallel (%arg0, %arg1) = (0, 0) to (32, 16) {
    %3 = memref.alloc() : memref<8x128xf32, 2>
    %4 = memref.alloc() : memref<8x128xf32, 2>
    %5 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
    %6 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg1)
    affine.parallel (%arg2, %arg3) = (0, 0) to (16, 16) {
      %7 = memref.alloc() : memref<4xf32, 3>
      %8 = memref.alloc() : memref<4xf32, 3>
      %9 = memref.alloc() : memref<8xf32, 3>
      %10 = memref.alloc() : memref<8xf32, 3>
      %11 = memref.alloc() : memref<8x8xf32, 3>
      %12 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg2)
      %13 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg3)
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %cst = arith.constant 0.000000e+00 : f32
          affine.store %cst, %11[%arg4, %arg5] : memref<8x8xf32, 3>
        }
      }
      affine.for %arg4 = 0 to 1024 step 8 {
        affine.for %arg5 = 0 to 1 {
          %14 = affine.vector_load %0[%arg2 * 8 + %arg3 floordiv 2 + %arg5 * 128 + %arg0 * 128, (%arg3 mod 2) * 4 + %arg4] : memref<4096x1024xf32, 1>, vector<4xf32>
          affine.vector_store %14, %7[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
        }
        affine.for %arg5 = 0 to 1 {
          %14 = affine.vector_load %1[(%arg2 * 16 + %arg3) floordiv 32 + %arg5 * 8 + %arg4, ((%arg2 * 16 + %arg3) mod 32) * 4 + %arg1 * 128] : memref<1024x2048xf32, 1>, vector<4xf32>
          affine.vector_store %14, %8[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
        }
        affine.for %arg5 = 0 to 8 {
          affine.for %arg6 = 0 to 8 {
            affine.for %arg7 = 0 to 8 {
              %14 = affine.load %11[%arg6, %arg7] : memref<8x8xf32, 3>
              %15 = affine.load %0[%5 + %12 + %arg6, %arg4 + %arg5] : memref<4096x1024xf32, 1>
              %16 = affine.load %1[%arg4 + %arg5, %6 + %13 + %arg7] : memref<1024x2048xf32, 1>
              %17 = arith.mulf %15, %16 : f32
              %18 = arith.addf %17, %14 : f32
              affine.store %18, %11[%arg6, %arg7] : memref<8x8xf32, 3>
            }
          }
        }
      }
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %14 = affine.load %11[%arg4, %arg5] : memref<8x8xf32, 3>
          affine.store %14, %2[%5 + %12 + %arg4, %6 + %13 + %arg5] : memref<4096x2048xf32, 1>
        }
      }
    }
  }
}
```
write
```c++
module @fuse_matmul_relu {
  %0 = memref.alloc() : memref<4096x1024xf32, 1>
  %1 = memref.alloc() : memref<1024x2048xf32, 1>
  %2 = memref.alloc() : memref<4096x2048xf32, 1>
  affine.parallel (%arg0, %arg1) = (0, 0) to (32, 16) {
    %3 = memref.alloc() : memref<8x128xf32, 2>
    %4 = memref.alloc() : memref<8x128xf32, 2>
    %5 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
    %6 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg1)
    affine.parallel (%arg2, %arg3) = (0, 0) to (16, 16) {
      %7 = memref.alloc() : memref<4xf32, 3>
      %8 = memref.alloc() : memref<4xf32, 3>
      %9 = memref.alloc() : memref<8xf32, 3>
      %10 = memref.alloc() : memref<8xf32, 3>
      %11 = memref.alloc() : memref<8x8xf32, 3>
      %12 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg2)
      %13 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg3)
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %cst = arith.constant 0.000000e+00 : f32
          affine.store %cst, %11[%arg4, %arg5] : memref<8x8xf32, 3>
        }
      }
      affine.for %arg4 = 0 to 1024 step 8 {
        gpu.barrier
        affine.for %arg5 = 0 to 1 {
          %14 = affine.vector_load %0[%arg2 * 8 + %arg3 floordiv 2 + %arg5 * 128 + %arg0 * 128, (%arg3 mod 2) * 4 + %arg4] : memref<4096x1024xf32, 1>, vector<4xf32>
          affine.vector_store %14, %7[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
        }
        affine.for %arg5 = 0 to 1 {
          %14 = affine.vector_load %1[(%arg2 * 16 + %arg3) floordiv 32 + %arg5 * 8 + %arg4, ((%arg2 * 16 + %arg3) mod 32) * 4 + %arg1 * 128] : memref<1024x2048xf32, 1>, vector<4xf32>
          affine.vector_store %14, %8[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
        }
        affine.for %arg5 = 0 to 1 {
          affine.for %arg6 = 0 to 4 {
            %14 = affine.vector_load %7[%arg5 * 4 + %arg6] : memref<4xf32, 3>, vector<1xf32>
            affine.vector_store %14, %3[(%arg3 mod 2) * 4 + %arg6, %arg2 * 8 + %arg3 floordiv 2 + %arg5 * 128] : memref<8x128xf32, 2>, vector<1xf32>
          }
        }
        affine.for %arg5 = 0 to 1 {
          %14 = affine.vector_load %8[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
          affine.vector_store %14, %4[(%arg2 * 16 + %arg3) floordiv 32 + %arg5 * 8, ((%arg2 * 16 + %arg3) mod 32) * 4] : memref<8x128xf32, 2>, vector<4xf32>
        }
        gpu.barrier
        affine.for %arg5 = 0 to 8 {
          affine.for %arg6 = 0 to 8 {
            affine.for %arg7 = 0 to 8 {
              %14 = affine.load %11[%arg6, %arg7] : memref<8x8xf32, 3>
              %15 = affine.load %0[%5 + %12 + %arg6, %arg4 + %arg5] : memref<4096x1024xf32, 1>
              %16 = affine.load %1[%arg4 + %arg5, %6 + %13 + %arg7] : memref<1024x2048xf32, 1>
              %17 = arith.mulf %15, %16 : f32
              %18 = arith.addf %17, %14 : f32
              affine.store %18, %11[%arg6, %arg7] : memref<8x8xf32, 3>
            }
          }
        }
      }
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %14 = affine.load %11[%arg4, %arg5] : memref<8x8xf32, 3>
          affine.store %14, %2[%5 + %12 + %arg4, %6 + %13 + %arg5] : memref<4096x2048xf32, 1>
        }
      }
    }
  }
}
```
read
```c++
module @fuse_matmul_relu {
  %0 = memref.alloc() : memref<4096x1024xf32, 1>
  %1 = memref.alloc() : memref<1024x2048xf32, 1>
  %2 = memref.alloc() : memref<4096x2048xf32, 1>
  affine.parallel (%arg0, %arg1) = (0, 0) to (32, 16) {
    %3 = memref.alloc() : memref<8x128xf32, 2>
    %4 = memref.alloc() : memref<8x128xf32, 2>
    %5 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
    %6 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg1)
    affine.parallel (%arg2, %arg3) = (0, 0) to (16, 16) {
      %7 = memref.alloc() : memref<4xf32, 3>
      %8 = memref.alloc() : memref<4xf32, 3>
      %9 = memref.alloc() : memref<8xf32, 3>
      %10 = memref.alloc() : memref<8xf32, 3>
      %11 = memref.alloc() : memref<8x8xf32, 3>
      %12 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg2)
      %13 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg3)
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %cst = arith.constant 0.000000e+00 : f32
          affine.store %cst, %11[%arg4, %arg5] : memref<8x8xf32, 3>
        }
      }
      affine.for %arg4 = 0 to 1024 step 8 {
        gpu.barrier
        affine.for %arg5 = 0 to 1 {
          %14 = affine.vector_load %0[%arg2 * 8 + %arg3 floordiv 2 + %arg5 * 128 + %arg0 * 128, (%arg3 mod 2) * 4 + %arg4] : memref<4096x1024xf32, 1>, vector<4xf32>
          affine.vector_store %14, %7[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
        }
        affine.for %arg5 = 0 to 1 {
          %14 = affine.vector_load %1[(%arg2 * 16 + %arg3) floordiv 32 + %arg5 * 8 + %arg4, ((%arg2 * 16 + %arg3) mod 32) * 4 + %arg1 * 128] : memref<1024x2048xf32, 1>, vector<4xf32>
          affine.vector_store %14, %8[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
        }
        affine.for %arg5 = 0 to 1 {
          affine.for %arg6 = 0 to 4 {
            %14 = affine.vector_load %7[%arg5 * 4 + %arg6] : memref<4xf32, 3>, vector<1xf32>
            affine.vector_store %14, %3[(%arg3 mod 2) * 4 + %arg6, %arg2 * 8 + %arg3 floordiv 2 + %arg5 * 128] : memref<8x128xf32, 2>, vector<1xf32>
          }
        }
        affine.for %arg5 = 0 to 1 {
          %14 = affine.vector_load %8[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
          affine.vector_store %14, %4[(%arg2 * 16 + %arg3) floordiv 32 + %arg5 * 8, ((%arg2 * 16 + %arg3) mod 32) * 4] : memref<8x128xf32, 2>, vector<4xf32>
        }
        gpu.barrier
        affine.for %arg5 = 0 to 8 {
          affine.for %arg6 = 0 to 2 {
            %14 = affine.vector_load %3[%arg5, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %arg6 * 2) * 8) * 4] : memref<8x128xf32, 2>, vector<4xf32>
            affine.vector_store %14, %9[%arg6 * 4] : memref<8xf32, 3>, vector<4xf32>
          }
          affine.for %arg6 = 0 to 2 {
            %14 = affine.vector_load %4[%arg5, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %arg6 * 4) * 4) * 4] : memref<8x128xf32, 2>, vector<4xf32>
            affine.vector_store %14, %10[%arg6 * 4] : memref<8xf32, 3>, vector<4xf32>
          }
          affine.for %arg6 = 0 to 8 {
            affine.for %arg7 = 0 to 8 {
              %14 = affine.load %11[%arg6, %arg7] : memref<8x8xf32, 3>
              %15 = affine.load %0[%5 + %12 + %arg6, %arg4 + %arg5] : memref<4096x1024xf32, 1>
              %16 = affine.load %1[%arg4 + %arg5, %6 + %13 + %arg7] : memref<1024x2048xf32, 1>
              %17 = arith.mulf %15, %16 : f32
              %18 = arith.addf %17, %14 : f32
              affine.store %18, %11[%arg6, %arg7] : memref<8x8xf32, 3>
            }
          }
        }
      }
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %14 = affine.load %11[%arg4, %arg5] : memref<8x8xf32, 3>
          affine.store %14, %2[%5 + %12 + %arg4, %6 + %13 + %arg5] : memref<4096x2048xf32, 1>
        }
      }
    }
  }
}
```
cache read
```c++
module @fuse_matmul_relu {
  %0 = memref.alloc() : memref<4096x1024xf32, 1>
  %1 = memref.alloc() : memref<1024x2048xf32, 1>
  %2 = memref.alloc() : memref<4096x2048xf32, 1>
  affine.parallel (%arg0, %arg1) = (0, 0) to (32, 16) {
    %3 = memref.alloc() : memref<8x128xf32, 2>
    %4 = memref.alloc() : memref<8x128xf32, 2>
    %5 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
    %6 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg1)
    affine.parallel (%arg2, %arg3) = (0, 0) to (16, 16) {
      %7 = memref.alloc() : memref<4xf32, 3>
      %8 = memref.alloc() : memref<4xf32, 3>
      %9 = memref.alloc() : memref<8xf32, 3>
      %10 = memref.alloc() : memref<8xf32, 3>
      %11 = memref.alloc() : memref<8x8xf32, 3>
      %12 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg2)
      %13 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg3)
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %cst = arith.constant 0.000000e+00 : f32
          affine.store %cst, %11[%arg4, %arg5] : memref<8x8xf32, 3>
        }
      }
      affine.for %arg4 = 0 to 1024 step 8 {
        gpu.barrier
        affine.for %arg5 = 0 to 1 {
          %14 = affine.vector_load %0[%arg2 * 8 + %arg3 floordiv 2 + %arg5 * 128 + %arg0 * 128, (%arg3 mod 2) * 4 + %arg4] : memref<4096x1024xf32, 1>, vector<4xf32>
          affine.vector_store %14, %7[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
        }
        affine.for %arg5 = 0 to 1 {
          %14 = affine.vector_load %1[(%arg2 * 16 + %arg3) floordiv 32 + %arg5 * 8 + %arg4, ((%arg2 * 16 + %arg3) mod 32) * 4 + %arg1 * 128] : memref<1024x2048xf32, 1>, vector<4xf32>
          affine.vector_store %14, %8[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
        }
        affine.for %arg5 = 0 to 1 {
          affine.for %arg6 = 0 to 4 {
            %14 = affine.vector_load %7[%arg5 * 4 + %arg6] : memref<4xf32, 3>, vector<1xf32>
            affine.vector_store %14, %3[(%arg3 mod 2) * 4 + %arg6, %arg2 * 8 + %arg3 floordiv 2 + %arg5 * 128] : memref<8x128xf32, 2>, vector<1xf32>
          }
        }
        affine.for %arg5 = 0 to 1 {
          %14 = affine.vector_load %8[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
          affine.vector_store %14, %4[(%arg2 * 16 + %arg3) floordiv 32 + %arg5 * 8, ((%arg2 * 16 + %arg3) mod 32) * 4] : memref<8x128xf32, 2>, vector<4xf32>
        }
        gpu.barrier
        affine.for %arg5 = 0 to 8 {
          affine.for %arg6 = 0 to 2 {
            %14 = affine.vector_load %3[%arg5, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %arg6 * 2) * 8) * 4] : memref<8x128xf32, 2>, vector<4xf32>
            affine.vector_store %14, %9[%arg6 * 4] : memref<8xf32, 3>, vector<4xf32>
          }
          affine.for %arg6 = 0 to 2 {
            %14 = affine.vector_load %4[%arg5, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %arg6 * 4) * 4) * 4] : memref<8x128xf32, 2>, vector<4xf32>
            affine.vector_store %14, %10[%arg6 * 4] : memref<8xf32, 3>, vector<4xf32>
          }
          affine.for %arg6 = 0 to 8 {
            affine.for %arg7 = 0 to 8 {
              %14 = affine.load %11[%arg6, %arg7] : memref<8x8xf32, 3>
              %15 = affine.load %9[%arg6] : memref<8xf32, 3>
              %16 = affine.load %10[%arg7] : memref<8xf32, 3>
              %17 = arith.mulf %15, %16 : f32
              %18 = arith.addf %17, %14 : f32
              affine.store %18, %11[%arg6, %arg7] : memref<8x8xf32, 3>
            }
          }
        }
      }
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %14 = affine.load %11[%arg4, %arg5] : memref<8x8xf32, 3>
          affine.store %14, %2[%5 + %12 + %arg4, %6 + %13 + %arg5] : memref<4096x2048xf32, 1>
        }
      }
    }
  }
}
```
cache write
```c++
module @fuse_matmul_relu {
  %0 = memref.alloc() : memref<4096x1024xf32, 1>
  %1 = memref.alloc() : memref<1024x2048xf32, 1>
  %2 = memref.alloc() : memref<4096x2048xf32, 1>
  affine.parallel (%arg0, %arg1) = (0, 0) to (32, 16) {
    %3 = memref.alloc() : memref<8x128xf32, 2>
    %4 = memref.alloc() : memref<8x128xf32, 2>
    %5 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
    %6 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg1)
    affine.parallel (%arg2, %arg3) = (0, 0) to (16, 16) {
      %7 = memref.alloc() : memref<4xf32, 3>
      %8 = memref.alloc() : memref<4xf32, 3>
      %9 = memref.alloc() : memref<8xf32, 3>
      %10 = memref.alloc() : memref<8xf32, 3>
      %11 = memref.alloc() : memref<8x8xf32, 3>
      %12 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg2)
      %13 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg3)
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %cst = arith.constant 0.000000e+00 : f32
          affine.store %cst, %11[%arg4, %arg5] : memref<8x8xf32, 3>
        }
      }
      affine.for %arg4 = 0 to 1024 step 8 {
        gpu.barrier
        affine.for %arg5 = 0 to 1 {
          %14 = affine.vector_load %0[%arg2 * 8 + %arg3 floordiv 2 + %arg5 * 128 + %arg0 * 128, (%arg3 mod 2) * 4 + %arg4] : memref<4096x1024xf32, 1>, vector<4xf32>
          affine.vector_store %14, %7[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
        }
        affine.for %arg5 = 0 to 1 {
          %14 = affine.vector_load %1[(%arg2 * 16 + %arg3) floordiv 32 + %arg5 * 8 + %arg4, ((%arg2 * 16 + %arg3) mod 32) * 4 + %arg1 * 128] : memref<1024x2048xf32, 1>, vector<4xf32>
          affine.vector_store %14, %8[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
        }
        affine.for %arg5 = 0 to 1 {
          affine.for %arg6 = 0 to 4 {
            %14 = affine.vector_load %7[%arg5 * 4 + %arg6] : memref<4xf32, 3>, vector<1xf32>
            affine.vector_store %14, %3[(%arg3 mod 2) * 4 + %arg6, %arg2 * 8 + %arg3 floordiv 2 + %arg5 * 128] : memref<8x128xf32, 2>, vector<1xf32>
          }
        }
        affine.for %arg5 = 0 to 1 {
          %14 = affine.vector_load %8[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
          affine.vector_store %14, %4[(%arg2 * 16 + %arg3) floordiv 32 + %arg5 * 8, ((%arg2 * 16 + %arg3) mod 32) * 4] : memref<8x128xf32, 2>, vector<4xf32>
        }
        gpu.barrier
        affine.for %arg5 = 0 to 8 {
          affine.for %arg6 = 0 to 2 {
            %14 = affine.vector_load %3[%arg5, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %arg6 * 2) * 8) * 4] : memref<8x128xf32, 2>, vector<4xf32>
            affine.vector_store %14, %9[%arg6 * 4] : memref<8xf32, 3>, vector<4xf32>
          }
          affine.for %arg6 = 0 to 2 {
            %14 = affine.vector_load %4[%arg5, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %arg6 * 4) * 4) * 4] : memref<8x128xf32, 2>, vector<4xf32>
            affine.vector_store %14, %10[%arg6 * 4] : memref<8xf32, 3>, vector<4xf32>
          }
          affine.for %arg6 = 0 to 8 {
            affine.for %arg7 = 0 to 8 {
              %14 = affine.load %11[%arg6, %arg7] : memref<8x8xf32, 3>
              %15 = affine.load %9[%arg6] : memref<8xf32, 3>
              %16 = affine.load %10[%arg7] : memref<8xf32, 3>
              %17 = arith.mulf %15, %16 : f32
              %18 = arith.addf %17, %14 : f32
              affine.store %18, %11[%arg6, %arg7] : memref<8x8xf32, 3>
            }
          }
        }
      }
      affine.for %arg4 = 0 to 8 step 4 {
        affine.for %arg5 = 0 to 8 step 4 {
          affine.for %arg6 = 0 to 4 {
            affine.for %arg7 = 0 to 4 {
              %14 = affine.load %11[%arg4 + %arg6, %arg5 + %arg7] : memref<8x8xf32, 3>
              affine.store %14, %2[%5 + %12 + %arg4 + %arg6, %6 + %13 + %arg5 + %arg7] : memref<4096x2048xf32, 1>
            }
          }
        }
      }
    }
  }
}
```
split + reorder
```c++
module @fuse_matmul_relu {
  %0 = memref.alloc() : memref<4096x1024xf32, 1>
  %1 = memref.alloc() : memref<1024x2048xf32, 1>
  %2 = memref.alloc() : memref<4096x2048xf32, 1>
  affine.parallel (%arg0, %arg1) = (0, 0) to (32, 16) {
    %3 = memref.alloc() : memref<8x128xf32, 2>
    %4 = memref.alloc() : memref<8x128xf32, 2>
    %5 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
    %6 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg1)
    affine.parallel (%arg2, %arg3) = (0, 0) to (16, 16) {
      %7 = memref.alloc() : memref<4xf32, 3>
      %8 = memref.alloc() : memref<4xf32, 3>
      %9 = memref.alloc() : memref<8xf32, 3>
      %10 = memref.alloc() : memref<8xf32, 3>
      %11 = memref.alloc() : memref<8x8xf32, 3>
      %12 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg2)
      %13 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg3)
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %cst = arith.constant 0.000000e+00 : f32
          affine.store %cst, %11[%arg4, %arg5] : memref<8x8xf32, 3>
        }
      }
      affine.for %arg4 = 0 to 1024 step 8 {
        gpu.barrier
        affine.for %arg5 = 0 to 1 {
          %14 = affine.vector_load %0[%arg2 * 8 + %arg3 floordiv 2 + %arg5 * 128 + %arg0 * 128, (%arg3 mod 2) * 4 + %arg4] : memref<4096x1024xf32, 1>, vector<4xf32>
          affine.vector_store %14, %7[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
        }
        affine.for %arg5 = 0 to 1 {
          %14 = affine.vector_load %1[(%arg2 * 16 + %arg3) floordiv 32 + %arg5 * 8 + %arg4, ((%arg2 * 16 + %arg3) mod 32) * 4 + %arg1 * 128] : memref<1024x2048xf32, 1>, vector<4xf32>
          affine.vector_store %14, %8[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
        }
        affine.for %arg5 = 0 to 1 {
          affine.for %arg6 = 0 to 4 {
            %14 = affine.vector_load %7[%arg5 * 4 + %arg6] : memref<4xf32, 3>, vector<1xf32>
            affine.vector_store %14, %3[(%arg3 mod 2) * 4 + %arg6, %arg2 * 8 + %arg3 floordiv 2 + %arg5 * 128] : memref<8x128xf32, 2>, vector<1xf32>
          }
        }
        affine.for %arg5 = 0 to 1 {
          %14 = affine.vector_load %8[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
          affine.vector_store %14, %4[(%arg2 * 16 + %arg3) floordiv 32 + %arg5 * 8, ((%arg2 * 16 + %arg3) mod 32) * 4] : memref<8x128xf32, 2>, vector<4xf32>
        }
        gpu.barrier
        affine.for %arg5 = 0 to 8 {
          affine.for %arg6 = 0 to 2 {
            %14 = affine.vector_load %3[%arg5, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %arg6 * 2) * 8) * 4] : memref<8x128xf32, 2>, vector<4xf32>
            affine.vector_store %14, %9[%arg6 * 4] : memref<8xf32, 3>, vector<4xf32>
          }
          affine.for %arg6 = 0 to 2 {
            %14 = affine.vector_load %4[%arg5, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %arg6 * 4) * 4) * 4] : memref<8x128xf32, 2>, vector<4xf32>
            affine.vector_store %14, %10[%arg6 * 4] : memref<8xf32, 3>, vector<4xf32>
          }
          affine.for %arg6 = 0 to 8 {
            affine.for %arg7 = 0 to 8 {
              %14 = affine.load %11[%arg6, %arg7] : memref<8x8xf32, 3>
              %15 = affine.load %9[%arg6] : memref<8xf32, 3>
              %16 = affine.load %10[%arg7] : memref<8xf32, 3>
              %17 = arith.mulf %15, %16 : f32
              %18 = arith.addf %17, %14 : f32
              affine.store %18, %11[%arg6, %arg7] : memref<8x8xf32, 3>
            }
          }
        }
      }
      affine.for %arg4 = 0 to 8 step 4 {
        affine.for %arg5 = 0 to 8 step 4 {
          affine.for %arg6 = 0 to 4 {
            affine.for %arg7 = 0 to 4 {
              %14 = affine.load %11[%arg4 + %arg6, %arg5 + %arg7] : memref<8x8xf32, 3>
              affine.store %14, %2[%arg6 + %arg4 * 16 + %arg2 * 4 + %arg0 * 128, %arg7 + %arg5 * 16 + %arg3 * 4 + %arg1 * 128] : memref<4096x2048xf32, 1>
            }
          }
        }
      }
    }
  }
}
```
vectorize
```c++
module @fuse_matmul_relu {
  %0 = memref.alloc() : memref<4096x1024xf32, 1>
  %1 = memref.alloc() : memref<1024x2048xf32, 1>
  %2 = memref.alloc() : memref<4096x2048xf32, 1>
  affine.parallel (%arg0, %arg1) = (0, 0) to (32, 16) {
    %3 = memref.alloc() : memref<8x128xf32, 2>
    %4 = memref.alloc() : memref<8x128xf32, 2>
    %5 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
    %6 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg1)
    affine.parallel (%arg2, %arg3) = (0, 0) to (16, 16) {
      %7 = memref.alloc() : memref<4xf32, 3>
      %8 = memref.alloc() : memref<4xf32, 3>
      %9 = memref.alloc() : memref<8xf32, 3>
      %10 = memref.alloc() : memref<8xf32, 3>
      %11 = memref.alloc() : memref<8x8xf32, 3>
      %12 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg2)
      %13 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg3)
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %cst = arith.constant 0.000000e+00 : f32
          affine.store %cst, %11[%arg4, %arg5] : memref<8x8xf32, 3>
        }
      }
      affine.for %arg4 = 0 to 1024 step 8 {
        gpu.barrier
        affine.for %arg5 = 0 to 1 {
          %14 = affine.vector_load %0[%arg2 * 8 + %arg3 floordiv 2 + %arg5 * 128 + %arg0 * 128, (%arg3 mod 2) * 4 + %arg4] : memref<4096x1024xf32, 1>, vector<4xf32>
          affine.vector_store %14, %7[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
        }
        affine.for %arg5 = 0 to 1 {
          %14 = affine.vector_load %1[(%arg2 * 16 + %arg3) floordiv 32 + %arg5 * 8 + %arg4, ((%arg2 * 16 + %arg3) mod 32) * 4 + %arg1 * 128] : memref<1024x2048xf32, 1>, vector<4xf32>
          affine.vector_store %14, %8[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
        }
        affine.for %arg5 = 0 to 1 {
          affine.for %arg6 = 0 to 4 {
            %14 = affine.vector_load %7[%arg5 * 4 + %arg6] : memref<4xf32, 3>, vector<1xf32>
            affine.vector_store %14, %3[(%arg3 mod 2) * 4 + %arg6, %arg2 * 8 + %arg3 floordiv 2 + %arg5 * 128] : memref<8x128xf32, 2>, vector<1xf32>
          }
        }
        affine.for %arg5 = 0 to 1 {
          %14 = affine.vector_load %8[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
          affine.vector_store %14, %4[(%arg2 * 16 + %arg3) floordiv 32 + %arg5 * 8, ((%arg2 * 16 + %arg3) mod 32) * 4] : memref<8x128xf32, 2>, vector<4xf32>
        }
        gpu.barrier
        affine.for %arg5 = 0 to 8 {
          affine.for %arg6 = 0 to 2 {
            %14 = affine.vector_load %3[%arg5, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %arg6 * 2) * 8) * 4] : memref<8x128xf32, 2>, vector<4xf32>
            affine.vector_store %14, %9[%arg6 * 4] : memref<8xf32, 3>, vector<4xf32>
          }
          affine.for %arg6 = 0 to 2 {
            %14 = affine.vector_load %4[%arg5, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %arg6 * 4) * 4) * 4] : memref<8x128xf32, 2>, vector<4xf32>
            affine.vector_store %14, %10[%arg6 * 4] : memref<8xf32, 3>, vector<4xf32>
          }
          affine.for %arg6 = 0 to 8 {
            affine.for %arg7 = 0 to 8 {
              %14 = affine.load %11[%arg6, %arg7] : memref<8x8xf32, 3>
              %15 = affine.load %9[%arg6] : memref<8xf32, 3>
              %16 = affine.load %10[%arg7] : memref<8xf32, 3>
              %17 = arith.mulf %15, %16 : f32
              %18 = arith.addf %17, %14 : f32
              affine.store %18, %11[%arg6, %arg7] : memref<8x8xf32, 3>
            }
          }
        }
      }
      affine.for %arg4 = 0 to 8 step 4 {
        affine.for %arg5 = 0 to 8 step 4 {
          affine.for %arg6 = 0 to 4 {
            affine.for %arg7 = 0 to 4 step 4 {
              %14 = affine.vector_load %11[%arg4 + %arg6, %arg5 + %arg7] : memref<8x8xf32, 3>, vector<4xf32>
              affine.vector_store %14, %2[%arg6 + %arg4 * 16 + %arg2 * 4 + %arg0 * 128, %arg7 + %arg5 * 16 + %arg3 * 4 + %arg1 * 128] : memref<4096x2048xf32, 1>, vector<4xf32>
            }
          }
        }
      }
    }
  }
}
```
pipeline
```c++
module @fuse_matmul_relu {
  %0 = memref.alloc() : memref<4096x1024xf32, 1>
  %1 = memref.alloc() : memref<1024x2048xf32, 1>
  %2 = memref.alloc() : memref<4096x2048xf32, 1>
  affine.parallel (%arg0, %arg1) = (0, 0) to (32, 16) {
    %3 = memref.alloc() : memref<2x8x128xf32, 2>
    %4 = memref.alloc() : memref<2x8x128xf32, 2>
    %5 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
    %6 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg1)
    affine.parallel (%arg2, %arg3) = (0, 0) to (16, 16) {
      %c0 = arith.constant 0 : index
      %c0_0 = arith.constant 0 : index
      %c0_1 = arith.constant 0 : index
      %c0_2 = arith.constant 0 : index
      %7 = memref.alloc() : memref<4xf32, 3>
      %8 = memref.alloc() : memref<4xf32, 3>
      %9 = memref.alloc() : memref<2x8xf32, 3>
      %10 = memref.alloc() : memref<2x8xf32, 3>
      %11 = memref.alloc() : memref<8x8xf32, 3>
      %12 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg2)
      %13 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg3)
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %cst = arith.constant 0.000000e+00 : f32
          affine.store %cst, %11[%arg4, %arg5] : memref<8x8xf32, 3>
        }
      }
      affine.for %arg4 = 0 to 1 {
        %14 = affine.vector_load %1[(%arg2 * 16 + %arg3) floordiv 32 + %arg4 * 8 + %c0_2, ((%arg2 * 16 + %arg3) mod 32) * 4 + %arg1 * 128] : memref<1024x2048xf32, 1>, vector<4xf32>
        affine.vector_store %14, %8[%arg4 * 4] : memref<4xf32, 3>, vector<4xf32>
      }
      affine.for %arg4 = 0 to 1 {
        %14 = affine.vector_load %8[%arg4 * 4] : memref<4xf32, 3>, vector<4xf32>
        affine.vector_store %14, %4[0, (%arg2 * 16 + %arg3) floordiv 32 + %arg4 * 8, ((%arg2 * 16 + %arg3) mod 32) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
      }
      affine.for %arg4 = 0 to 1 {
        %14 = affine.vector_load %0[%arg2 * 8 + %arg3 floordiv 2 + %arg4 * 128 + %arg0 * 128, (%arg3 mod 2) * 4 + %c0_1] : memref<4096x1024xf32, 1>, vector<4xf32>
        affine.vector_store %14, %7[%arg4 * 4] : memref<4xf32, 3>, vector<4xf32>
      }
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 4 {
          %14 = affine.vector_load %7[%arg4 * 4 + %arg5] : memref<4xf32, 3>, vector<1xf32>
          affine.vector_store %14, %3[0, (%arg3 mod 2) * 4 + %arg5, %arg2 * 8 + %arg3 floordiv 2 + %arg4 * 128] : memref<2x8x128xf32, 2>, vector<1xf32>
        }
      }
      affine.for %arg4 = 0 to 1024 step 8 {
        affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg4) {
          affine.for %arg5 = 0 to 1 {
            %14 = affine.vector_load %0[%arg2 * 8 + %arg3 floordiv 2 + %arg5 * 128 + %arg0 * 128, (%arg3 mod 2) * 4 + %arg4 + 8] : memref<4096x1024xf32, 1>, vector<4xf32>
            affine.vector_store %14, %7[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
          }
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4 {
              %14 = affine.vector_load %7[%arg5 * 4 + %arg6] : memref<4xf32, 3>, vector<1xf32>
              affine.vector_store %14, %3[(%arg4 floordiv 8 + 1) mod 2, (%arg3 mod 2) * 4 + %arg6, %arg2 * 8 + %arg3 floordiv 2 + %arg5 * 128] : memref<2x8x128xf32, 2>, vector<1xf32>
            }
          }
        }
        affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg4) {
          affine.for %arg5 = 0 to 1 {
            %14 = affine.vector_load %1[(%arg2 * 16 + %arg3) floordiv 32 + %arg5 * 8 + %arg4 + 8, ((%arg2 * 16 + %arg3) mod 32) * 4 + %arg1 * 128] : memref<1024x2048xf32, 1>, vector<4xf32>
            affine.vector_store %14, %8[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
          }
          affine.for %arg5 = 0 to 1 {
            %14 = affine.vector_load %8[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
            affine.vector_store %14, %4[(%arg4 floordiv 8 + 1) mod 2, (%arg2 * 16 + %arg3) floordiv 32 + %arg5 * 8, ((%arg2 * 16 + %arg3) mod 32) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          }
        }
        gpu.barrier
        gpu.barrier
        affine.for %arg5 = 0 to 2 {
          %14 = affine.vector_load %4[(%arg4 floordiv 8) mod 2, %c0_0, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %arg5 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %14, %10[0, %arg5 * 4] : memref<2x8xf32, 3>, vector<4xf32>
        }
        affine.for %arg5 = 0 to 2 {
          %14 = affine.vector_load %3[(%arg4 floordiv 8) mod 2, %c0, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %arg5 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %14, %9[0, %arg5 * 4] : memref<2x8xf32, 3>, vector<4xf32>
        }
        affine.for %arg5 = 0 to 8 {
          affine.if affine_set<(d0) : (-d0 + 6 >= 0)>(%arg5) {
            affine.for %arg6 = 0 to 2 {
              %14 = affine.vector_load %3[(%arg4 floordiv 8) mod 2, %arg5 + 1, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %arg6 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
              affine.vector_store %14, %9[(%arg5 + 1) mod 2, %arg6 * 4] : memref<2x8xf32, 3>, vector<4xf32>
            }
          }
          affine.if affine_set<(d0) : (-d0 + 6 >= 0)>(%arg5) {
            affine.for %arg6 = 0 to 2 {
              %14 = affine.vector_load %4[(%arg4 floordiv 8) mod 2, %arg5 + 1, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %arg6 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
              affine.vector_store %14, %10[(%arg5 + 1) mod 2, %arg6 * 4] : memref<2x8xf32, 3>, vector<4xf32>
            }
          }
          affine.for %arg6 = 0 to 8 {
            affine.for %arg7 = 0 to 8 {
              %14 = affine.load %11[%arg6, %arg7] : memref<8x8xf32, 3>
              %15 = affine.load %9[%arg5 mod 2, %arg6] : memref<2x8xf32, 3>
              %16 = affine.load %10[%arg5 mod 2, %arg7] : memref<2x8xf32, 3>
              %17 = arith.mulf %15, %16 : f32
              %18 = arith.addf %17, %14 : f32
              affine.store %18, %11[%arg6, %arg7] : memref<8x8xf32, 3>
            }
          }
        }
      }
      affine.for %arg4 = 0 to 8 step 4 {
        affine.for %arg5 = 0 to 8 step 4 {
          affine.for %arg6 = 0 to 4 {
            affine.for %arg7 = 0 to 4 step 4 {
              %14 = affine.vector_load %11[%arg4 + %arg6, %arg5 + %arg7] : memref<8x8xf32, 3>, vector<4xf32>
              affine.vector_store %14, %2[%arg6 + %arg4 * 16 + %arg2 * 4 + %arg0 * 128, %arg7 + %arg5 * 16 + %arg3 * 4 + %arg1 * 128] : memref<4096x2048xf32, 1>, vector<4xf32>
            }
          }
        }
      }
    }
  }
}
```
detach last loop
```c++
module @fuse_matmul_relu {
  %0 = memref.alloc() : memref<4096x1024xf32, 1>
  %1 = memref.alloc() : memref<1024x2048xf32, 1>
  %2 = memref.alloc() : memref<4096x2048xf32, 1>
  affine.parallel (%arg0, %arg1) = (0, 0) to (32, 16) {
    %3 = memref.alloc() : memref<2x8x128xf32, 2>
    %4 = memref.alloc() : memref<2x8x128xf32, 2>
    %5 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
    %6 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg1)
    affine.parallel (%arg2, %arg3) = (0, 0) to (16, 16) {
      %c7 = arith.constant 7 : index
      %c0 = arith.constant 0 : index
      %c0_0 = arith.constant 0 : index
      %c0_1 = arith.constant 0 : index
      %c0_2 = arith.constant 0 : index
      %7 = memref.alloc() : memref<4xf32, 3>
      %8 = memref.alloc() : memref<4xf32, 3>
      %9 = memref.alloc() : memref<2x8xf32, 3>
      %10 = memref.alloc() : memref<2x8xf32, 3>
      %11 = memref.alloc() : memref<8x8xf32, 3>
      %12 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg2)
      %13 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg3)
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %cst = arith.constant 0.000000e+00 : f32
          affine.store %cst, %11[%arg4, %arg5] : memref<8x8xf32, 3>
        }
      }
      affine.for %arg4 = 0 to 1 {
        %14 = affine.vector_load %1[(%arg2 * 16 + %arg3) floordiv 32 + %arg4 * 8 + %c0_2, ((%arg2 * 16 + %arg3) mod 32) * 4 + %arg1 * 128] : memref<1024x2048xf32, 1>, vector<4xf32>
        affine.vector_store %14, %8[%arg4 * 4] : memref<4xf32, 3>, vector<4xf32>
      }
      affine.for %arg4 = 0 to 1 {
        %14 = affine.vector_load %8[%arg4 * 4] : memref<4xf32, 3>, vector<4xf32>
        affine.vector_store %14, %4[0, (%arg2 * 16 + %arg3) floordiv 32 + %arg4 * 8, ((%arg2 * 16 + %arg3) mod 32) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
      }
      affine.for %arg4 = 0 to 1 {
        %14 = affine.vector_load %0[%arg2 * 8 + %arg3 floordiv 2 + %arg4 * 128 + %arg0 * 128, (%arg3 mod 2) * 4 + %c0_1] : memref<4096x1024xf32, 1>, vector<4xf32>
        affine.vector_store %14, %7[%arg4 * 4] : memref<4xf32, 3>, vector<4xf32>
      }
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 4 {
          %14 = affine.vector_load %7[%arg4 * 4 + %arg5] : memref<4xf32, 3>, vector<1xf32>
          affine.vector_store %14, %3[0, (%arg3 mod 2) * 4 + %arg5, %arg2 * 8 + %arg3 floordiv 2 + %arg4 * 128] : memref<2x8x128xf32, 2>, vector<1xf32>
        }
      }
      affine.for %arg4 = 0 to 1024 step 8 {
        affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg4) {
          affine.for %arg5 = 0 to 1 {
            %14 = affine.vector_load %0[%arg2 * 8 + %arg3 floordiv 2 + %arg5 * 128 + %arg0 * 128, (%arg3 mod 2) * 4 + %arg4 + 8] : memref<4096x1024xf32, 1>, vector<4xf32>
            affine.vector_store %14, %7[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
          }
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4 {
              %14 = affine.vector_load %7[%arg5 * 4 + %arg6] : memref<4xf32, 3>, vector<1xf32>
              affine.vector_store %14, %3[(%arg4 floordiv 8 + 1) mod 2, (%arg3 mod 2) * 4 + %arg6, %arg2 * 8 + %arg3 floordiv 2 + %arg5 * 128] : memref<2x8x128xf32, 2>, vector<1xf32>
            }
          }
        }
        affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg4) {
          affine.for %arg5 = 0 to 1 {
            %14 = affine.vector_load %1[(%arg2 * 16 + %arg3) floordiv 32 + %arg5 * 8 + %arg4 + 8, ((%arg2 * 16 + %arg3) mod 32) * 4 + %arg1 * 128] : memref<1024x2048xf32, 1>, vector<4xf32>
            affine.vector_store %14, %8[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
          }
          affine.for %arg5 = 0 to 1 {
            %14 = affine.vector_load %8[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
            affine.vector_store %14, %4[(%arg4 floordiv 8 + 1) mod 2, (%arg2 * 16 + %arg3) floordiv 32 + %arg5 * 8, ((%arg2 * 16 + %arg3) mod 32) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          }
        }
        gpu.barrier
        gpu.barrier
        affine.for %arg5 = 0 to 2 {
          %14 = affine.vector_load %4[(%arg4 floordiv 8) mod 2, %c0_0, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %arg5 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %14, %10[0, %arg5 * 4] : memref<2x8xf32, 3>, vector<4xf32>
        }
        affine.for %arg5 = 0 to 2 {
          %14 = affine.vector_load %3[(%arg4 floordiv 8) mod 2, %c0, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %arg5 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %14, %9[0, %arg5 * 4] : memref<2x8xf32, 3>, vector<4xf32>
        }
        affine.for %arg5 = 0 to 7 {
          affine.if affine_set<(d0) : (-d0 + 6 >= 0)>(%arg5) {
            affine.for %arg6 = 0 to 2 {
              %14 = affine.vector_load %3[(%arg4 floordiv 8) mod 2, %arg5 + 1, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %arg6 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
              affine.vector_store %14, %9[(%arg5 + 1) mod 2, %arg6 * 4] : memref<2x8xf32, 3>, vector<4xf32>
            }
          }
          affine.if affine_set<(d0) : (-d0 + 6 >= 0)>(%arg5) {
            affine.for %arg6 = 0 to 2 {
              %14 = affine.vector_load %4[(%arg4 floordiv 8) mod 2, %arg5 + 1, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %arg6 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
              affine.vector_store %14, %10[(%arg5 + 1) mod 2, %arg6 * 4] : memref<2x8xf32, 3>, vector<4xf32>
            }
          }
          affine.for %arg6 = 0 to 8 {
            affine.for %arg7 = 0 to 8 {
              %14 = affine.load %11[%arg6, %arg7] : memref<8x8xf32, 3>
              %15 = affine.load %9[%arg5 mod 2, %arg6] : memref<2x8xf32, 3>
              %16 = affine.load %10[%arg5 mod 2, %arg7] : memref<2x8xf32, 3>
              %17 = arith.mulf %15, %16 : f32
              %18 = arith.addf %17, %14 : f32
              affine.store %18, %11[%arg6, %arg7] : memref<8x8xf32, 3>
            }
          }
        }
        affine.if affine_set<(d0) : (-d0 + 6 >= 0)>(%c7) {
          affine.for %arg5 = 0 to 2 {
            %14 = affine.vector_load %3[(%arg4 floordiv 8) mod 2, %c7 + 1, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %arg5 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
            affine.vector_store %14, %9[(%c7 + 1) mod 2, %arg5 * 4] : memref<2x8xf32, 3>, vector<4xf32>
          }
        }
        affine.if affine_set<(d0) : (-d0 + 6 >= 0)>(%c7) {
          affine.for %arg5 = 0 to 2 {
            %14 = affine.vector_load %4[(%arg4 floordiv 8) mod 2, %c7 + 1, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %arg5 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
            affine.vector_store %14, %10[(%c7 + 1) mod 2, %arg5 * 4] : memref<2x8xf32, 3>, vector<4xf32>
          }
        }
        affine.for %arg5 = 0 to 8 {
          affine.for %arg6 = 0 to 8 {
            %14 = affine.load %11[%arg5, %arg6] : memref<8x8xf32, 3>
            %15 = affine.load %9[%c7 mod 2, %arg5] : memref<2x8xf32, 3>
            %16 = affine.load %10[%c7 mod 2, %arg6] : memref<2x8xf32, 3>
            %17 = arith.mulf %15, %16 : f32
            %18 = arith.addf %17, %14 : f32
            affine.store %18, %11[%arg5, %arg6] : memref<8x8xf32, 3>
          }
        }
      }
      affine.for %arg4 = 0 to 8 step 4 {
        affine.for %arg5 = 0 to 8 step 4 {
          affine.for %arg6 = 0 to 4 {
            affine.for %arg7 = 0 to 4 step 4 {
              %14 = affine.vector_load %11[%arg4 + %arg6, %arg5 + %arg7] : memref<8x8xf32, 3>, vector<4xf32>
              affine.vector_store %14, %2[%arg6 + %arg4 * 16 + %arg2 * 4 + %arg0 * 128, %arg7 + %arg5 * 16 + %arg3 * 4 + %arg1 * 128] : memref<4096x2048xf32, 1>, vector<4xf32>
            }
          }
        }
      }
    }
  }
}
```
schedule
```c++
module @fuse_matmul_relu {
  %0 = memref.alloc() : memref<4096x1024xf32, 1>
  %1 = memref.alloc() : memref<1024x2048xf32, 1>
  %2 = memref.alloc() : memref<4096x2048xf32, 1>
  affine.parallel (%arg0, %arg1) = (0, 0) to (32, 16) {
    %3 = memref.alloc() : memref<2x8x128xf32, 2>
    %4 = memref.alloc() : memref<2x8x128xf32, 2>
    %5 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
    %6 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg1)
    affine.parallel (%arg2, %arg3) = (0, 0) to (16, 16) {
      %c0 = arith.constant 0 : index
      %c0_0 = arith.constant 0 : index
      %c7 = arith.constant 7 : index
      %c0_1 = arith.constant 0 : index
      %c0_2 = arith.constant 0 : index
      %c0_3 = arith.constant 0 : index
      %c0_4 = arith.constant 0 : index
      %7 = memref.alloc() : memref<4xf32, 3>
      %8 = memref.alloc() : memref<4xf32, 3>
      %9 = memref.alloc() : memref<2x8xf32, 3>
      %10 = memref.alloc() : memref<2x8xf32, 3>
      %11 = memref.alloc() : memref<8x8xf32, 3>
      %12 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg2)
      %13 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg3)
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %cst = arith.constant 0.000000e+00 : f32
          affine.store %cst, %11[%arg4, %arg5] : memref<8x8xf32, 3>
        }
      }
      affine.for %arg4 = 0 to 1 {
        %14 = affine.vector_load %0[%arg2 * 8 + %arg3 floordiv 2 + %arg4 * 128 + %arg0 * 128, (%arg3 mod 2) * 4 + %c0_3] : memref<4096x1024xf32, 1>, vector<4xf32>
        affine.vector_store %14, %7[%arg4 * 4] : memref<4xf32, 3>, vector<4xf32>
      }
      affine.for %arg4 = 0 to 1 {
        %14 = affine.vector_load %1[(%arg2 * 16 + %arg3) floordiv 32 + %arg4 * 8 + %c0_4, ((%arg2 * 16 + %arg3) mod 32) * 4 + %arg1 * 128] : memref<1024x2048xf32, 1>, vector<4xf32>
        affine.vector_store %14, %8[%arg4 * 4] : memref<4xf32, 3>, vector<4xf32>
      }
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 4 {
          %14 = affine.vector_load %7[%arg4 * 4 + %arg5] : memref<4xf32, 3>, vector<1xf32>
          affine.vector_store %14, %3[0, (%arg3 mod 2) * 4 + %arg5, %arg2 * 8 + %arg3 floordiv 2 + %arg4 * 128] : memref<2x8x128xf32, 2>, vector<1xf32>
        }
      }
      affine.for %arg4 = 0 to 1 {
        %14 = affine.vector_load %8[%arg4 * 4] : memref<4xf32, 3>, vector<4xf32>
        affine.vector_store %14, %4[0, (%arg2 * 16 + %arg3) floordiv 32 + %arg4 * 8, ((%arg2 * 16 + %arg3) mod 32) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
      }
      gpu.barrier
      affine.for %arg4 = 0 to 2 {
        %14 = affine.vector_load %3[(%c0_0 floordiv 8) mod 2, %c0_1, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %arg4 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
        affine.vector_store %14, %9[0, %arg4 * 4] : memref<2x8xf32, 3>, vector<4xf32>
      }
      affine.for %arg4 = 0 to 2 {
        %14 = affine.vector_load %4[(%c0 floordiv 8) mod 2, %c0_2, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %arg4 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
        affine.vector_store %14, %10[0, %arg4 * 4] : memref<2x8xf32, 3>, vector<4xf32>
      }
      affine.for %arg4 = 0 to 1024 step 8 {
        affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg4) {
          affine.for %arg5 = 0 to 1 {
            %14 = affine.vector_load %0[%arg2 * 8 + %arg3 floordiv 2 + %arg5 * 128 + %arg0 * 128, (%arg3 mod 2) * 4 + %arg4 + 8] : memref<4096x1024xf32, 1>, vector<4xf32>
            affine.vector_store %14, %7[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
          }
          affine.for %arg5 = 0 to 1 {
            %14 = affine.vector_load %1[(%arg2 * 16 + %arg3) floordiv 32 + %arg5 * 8 + %arg4 + 8, ((%arg2 * 16 + %arg3) mod 32) * 4 + %arg1 * 128] : memref<1024x2048xf32, 1>, vector<4xf32>
            affine.vector_store %14, %8[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
          }
        }
        affine.for %arg5 = 0 to 7 {
          affine.if affine_set<(d0) : (-d0 + 6 >= 0)>(%arg5) {
            affine.for %arg6 = 0 to 2 {
              %14 = affine.vector_load %3[(%arg4 floordiv 8) mod 2, %arg5 + 1, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %arg6 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
              affine.vector_store %14, %9[(%arg5 + 1) mod 2, %arg6 * 4] : memref<2x8xf32, 3>, vector<4xf32>
            }
          }
          affine.if affine_set<(d0) : (-d0 + 6 >= 0)>(%arg5) {
            affine.for %arg6 = 0 to 2 {
              %14 = affine.vector_load %4[(%arg4 floordiv 8) mod 2, %arg5 + 1, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %arg6 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
              affine.vector_store %14, %10[(%arg5 + 1) mod 2, %arg6 * 4] : memref<2x8xf32, 3>, vector<4xf32>
            }
          }
          affine.for %arg6 = 0 to 8 {
            affine.for %arg7 = 0 to 8 {
              %14 = affine.load %11[%arg6, %arg7] : memref<8x8xf32, 3>
              %15 = affine.load %9[%arg5 mod 2, %arg6] : memref<2x8xf32, 3>
              %16 = affine.load %10[%arg5 mod 2, %arg7] : memref<2x8xf32, 3>
              %17 = arith.mulf %15, %16 : f32
              %18 = arith.addf %17, %14 : f32
              affine.store %18, %11[%arg6, %arg7] : memref<8x8xf32, 3>
            }
          }
        }
        affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg4) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4 {
              %14 = affine.vector_load %7[%arg5 * 4 + %arg6] : memref<4xf32, 3>, vector<1xf32>
              affine.vector_store %14, %3[(%arg4 floordiv 8 + 1) mod 2, (%arg3 mod 2) * 4 + %arg6, %arg2 * 8 + %arg3 floordiv 2 + %arg5 * 128] : memref<2x8x128xf32, 2>, vector<1xf32>
            }
          }
          affine.for %arg5 = 0 to 1 {
            %14 = affine.vector_load %8[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
            affine.vector_store %14, %4[(%arg4 floordiv 8 + 1) mod 2, (%arg2 * 16 + %arg3) floordiv 32 + %arg5 * 8, ((%arg2 * 16 + %arg3) mod 32) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          }
          gpu.barrier
        }
        affine.if affine_set<(d0) : (-d0 + 6 >= 0)>(%c7) {
          affine.for %arg5 = 0 to 2 {
            %14 = affine.vector_load %3[(%arg4 floordiv 8) mod 2, %c7 + 1, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %arg5 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
            affine.vector_store %14, %9[(%c7 + 1) mod 2, %arg5 * 4] : memref<2x8xf32, 3>, vector<4xf32>
          }
        }
        affine.if affine_set<(d0) : (-d0 + 6 >= 0)>(%c7) {
          affine.for %arg5 = 0 to 2 {
            %14 = affine.vector_load %4[(%arg4 floordiv 8) mod 2, %c7 + 1, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %arg5 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
            affine.vector_store %14, %10[(%c7 + 1) mod 2, %arg5 * 4] : memref<2x8xf32, 3>, vector<4xf32>
          }
        }
        affine.for %arg5 = 0 to 8 {
          affine.for %arg6 = 0 to 8 {
            %14 = affine.load %11[%arg5, %arg6] : memref<8x8xf32, 3>
            %15 = affine.load %9[%c7 mod 2, %arg5] : memref<2x8xf32, 3>
            %16 = affine.load %10[%c7 mod 2, %arg6] : memref<2x8xf32, 3>
            %17 = arith.mulf %15, %16 : f32
            %18 = arith.addf %17, %14 : f32
            affine.store %18, %11[%arg5, %arg6] : memref<8x8xf32, 3>
          }
        }
        affine.for %arg5 = 0 to 2 {
          %14 = affine.vector_load %4[(%arg4 floordiv 8) mod 2, %c0_2, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %arg5 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %14, %10[0, %arg5 * 4] : memref<2x8xf32, 3>, vector<4xf32>
        }
        affine.for %arg5 = 0 to 2 {
          %14 = affine.vector_load %3[(%arg4 floordiv 8) mod 2, %c0_1, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %arg5 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %14, %9[0, %arg5 * 4] : memref<2x8xf32, 3>, vector<4xf32>
        }
      }
      affine.for %arg4 = 0 to 8 step 4 {
        affine.for %arg5 = 0 to 8 step 4 {
          affine.for %arg6 = 0 to 4 {
            affine.for %arg7 = 0 to 4 step 4 {
              %14 = affine.vector_load %11[%arg4 + %arg6, %arg5 + %arg7] : memref<8x8xf32, 3>, vector<4xf32>
              affine.vector_store %14, %2[%arg6 + %arg4 * 16 + %arg2 * 4 + %arg0 * 128, %arg7 + %arg5 * 16 + %arg3 * 4 + %arg1 * 128] : memref<4096x2048xf32, 1>, vector<4xf32>
            }
          }
        }
      }
    }
  }
}
```
optimize `if`
```c++
module @fuse_matmul_relu {
  %0 = memref.alloc() : memref<4096x1024xf32, 1>
  %1 = memref.alloc() : memref<1024x2048xf32, 1>
  %2 = memref.alloc() : memref<4096x2048xf32, 1>
  affine.parallel (%arg0, %arg1) = (0, 0) to (32, 16) {
    %3 = memref.alloc() : memref<2x8x128xf32, 2>
    %4 = memref.alloc() : memref<2x8x128xf32, 2>
    %5 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
    %6 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg1)
    affine.parallel (%arg2, %arg3) = (0, 0) to (16, 16) {
      %c0 = arith.constant 0 : index
      %c0_0 = arith.constant 0 : index
      %c7 = arith.constant 7 : index
      %c0_1 = arith.constant 0 : index
      %c0_2 = arith.constant 0 : index
      %c0_3 = arith.constant 0 : index
      %c0_4 = arith.constant 0 : index
      %7 = memref.alloc() : memref<4xf32, 3>
      %8 = memref.alloc() : memref<4xf32, 3>
      %9 = memref.alloc() : memref<2x8xf32, 3>
      %10 = memref.alloc() : memref<2x8xf32, 3>
      %11 = memref.alloc() : memref<8x8xf32, 3>
      %12 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg2)
      %13 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg3)
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %cst = arith.constant 0.000000e+00 : f32
          affine.store %cst, %11[%arg4, %arg5] : memref<8x8xf32, 3>
        }
      }
      affine.for %arg4 = 0 to 1 {
        %14 = affine.vector_load %0[%arg2 * 8 + %arg3 floordiv 2 + %arg4 * 128 + %arg0 * 128, (%arg3 mod 2) * 4 + %c0_3] : memref<4096x1024xf32, 1>, vector<4xf32>
        affine.vector_store %14, %7[%arg4 * 4] : memref<4xf32, 3>, vector<4xf32>
      }
      affine.for %arg4 = 0 to 1 {
        %14 = affine.vector_load %1[(%arg2 * 16 + %arg3) floordiv 32 + %arg4 * 8 + %c0_4, ((%arg2 * 16 + %arg3) mod 32) * 4 + %arg1 * 128] : memref<1024x2048xf32, 1>, vector<4xf32>
        affine.vector_store %14, %8[%arg4 * 4] : memref<4xf32, 3>, vector<4xf32>
      }
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 4 {
          %14 = affine.vector_load %7[%arg4 * 4 + %arg5] : memref<4xf32, 3>, vector<1xf32>
          affine.vector_store %14, %3[0, (%arg3 mod 2) * 4 + %arg5, %arg2 * 8 + %arg3 floordiv 2 + %arg4 * 128] : memref<2x8x128xf32, 2>, vector<1xf32>
        }
      }
      affine.for %arg4 = 0 to 1 {
        %14 = affine.vector_load %8[%arg4 * 4] : memref<4xf32, 3>, vector<4xf32>
        affine.vector_store %14, %4[0, (%arg2 * 16 + %arg3) floordiv 32 + %arg4 * 8, ((%arg2 * 16 + %arg3) mod 32) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
      }
      gpu.barrier
      affine.for %arg4 = 0 to 2 {
        %14 = affine.vector_load %3[(%c0_0 floordiv 8) mod 2, %c0_1, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %arg4 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
        affine.vector_store %14, %9[0, %arg4 * 4] : memref<2x8xf32, 3>, vector<4xf32>
      }
      affine.for %arg4 = 0 to 2 {
        %14 = affine.vector_load %4[(%c0 floordiv 8) mod 2, %c0_2, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %arg4 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
        affine.vector_store %14, %10[0, %arg4 * 4] : memref<2x8xf32, 3>, vector<4xf32>
      }
      affine.for %arg4 = 0 to 1024 step 8 {
        affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg4) {
          affine.for %arg5 = 0 to 1 {
            %14 = affine.vector_load %0[%arg2 * 8 + %arg3 floordiv 2 + %arg5 * 128 + %arg0 * 128, (%arg3 mod 2) * 4 + %arg4 + 8] : memref<4096x1024xf32, 1>, vector<4xf32>
            affine.vector_store %14, %7[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
          }
          affine.for %arg5 = 0 to 1 {
            %14 = affine.vector_load %1[(%arg2 * 16 + %arg3) floordiv 32 + %arg5 * 8 + %arg4 + 8, ((%arg2 * 16 + %arg3) mod 32) * 4 + %arg1 * 128] : memref<1024x2048xf32, 1>, vector<4xf32>
            affine.vector_store %14, %8[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
          }
        }
        affine.for %arg5 = 0 to 7 {
          affine.for %arg6 = 0 to 2 {
            %14 = affine.vector_load %3[(%arg4 floordiv 8) mod 2, %arg5 + 1, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %arg6 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
            affine.vector_store %14, %9[(%arg5 + 1) mod 2, %arg6 * 4] : memref<2x8xf32, 3>, vector<4xf32>
          }
          affine.for %arg6 = 0 to 2 {
            %14 = affine.vector_load %4[(%arg4 floordiv 8) mod 2, %arg5 + 1, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %arg6 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
            affine.vector_store %14, %10[(%arg5 + 1) mod 2, %arg6 * 4] : memref<2x8xf32, 3>, vector<4xf32>
          }
          affine.for %arg6 = 0 to 8 {
            affine.for %arg7 = 0 to 8 {
              %14 = affine.load %11[%arg6, %arg7] : memref<8x8xf32, 3>
              %15 = affine.load %9[%arg5 mod 2, %arg6] : memref<2x8xf32, 3>
              %16 = affine.load %10[%arg5 mod 2, %arg7] : memref<2x8xf32, 3>
              %17 = arith.mulf %15, %16 : f32
              %18 = arith.addf %17, %14 : f32
              affine.store %18, %11[%arg6, %arg7] : memref<8x8xf32, 3>
            }
          }
        }
        affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg4) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4 {
              %14 = affine.vector_load %7[%arg5 * 4 + %arg6] : memref<4xf32, 3>, vector<1xf32>
              affine.vector_store %14, %3[(%arg4 floordiv 8 + 1) mod 2, (%arg3 mod 2) * 4 + %arg6, %arg2 * 8 + %arg3 floordiv 2 + %arg5 * 128] : memref<2x8x128xf32, 2>, vector<1xf32>
            }
          }
          affine.for %arg5 = 0 to 1 {
            %14 = affine.vector_load %8[%arg5 * 4] : memref<4xf32, 3>, vector<4xf32>
            affine.vector_store %14, %4[(%arg4 floordiv 8 + 1) mod 2, (%arg2 * 16 + %arg3) floordiv 32 + %arg5 * 8, ((%arg2 * 16 + %arg3) mod 32) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          }
          gpu.barrier
        }
        affine.for %arg5 = 0 to 8 {
          affine.for %arg6 = 0 to 8 {
            %14 = affine.load %11[%arg5, %arg6] : memref<8x8xf32, 3>
            %15 = affine.load %9[%c7 mod 2, %arg5] : memref<2x8xf32, 3>
            %16 = affine.load %10[%c7 mod 2, %arg6] : memref<2x8xf32, 3>
            %17 = arith.mulf %15, %16 : f32
            %18 = arith.addf %17, %14 : f32
            affine.store %18, %11[%arg5, %arg6] : memref<8x8xf32, 3>
          }
        }
        affine.for %arg5 = 0 to 2 {
          %14 = affine.vector_load %4[(%arg4 floordiv 8) mod 2, %c0_2, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %arg5 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %14, %10[0, %arg5 * 4] : memref<2x8xf32, 3>, vector<4xf32>
        }
        affine.for %arg5 = 0 to 2 {
          %14 = affine.vector_load %3[(%arg4 floordiv 8) mod 2, %c0_1, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %arg5 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %14, %9[0, %arg5 * 4] : memref<2x8xf32, 3>, vector<4xf32>
        }
      }
      affine.for %arg4 = 0 to 8 step 4 {
        affine.for %arg5 = 0 to 8 step 4 {
          affine.for %arg6 = 0 to 4 {
            affine.for %arg7 = 0 to 4 step 4 {
              %14 = affine.vector_load %11[%arg4 + %arg6, %arg5 + %arg7] : memref<8x8xf32, 3>, vector<4xf32>
              affine.vector_store %14, %2[%arg6 + %arg4 * 16 + %arg2 * 4 + %arg0 * 128, %arg7 + %arg5 * 16 + %arg3 * 4 + %arg1 * 128] : memref<4096x2048xf32, 1>, vector<4xf32>
            }
          }
        }
      }
    }
  }
}
```
loop unroll
```c++
module @fuse_matmul_relu {
  %0 = memref.alloc() : memref<4096x1024xf32, 1>
  %1 = memref.alloc() : memref<1024x2048xf32, 1>
  %2 = memref.alloc() : memref<4096x2048xf32, 1>
  affine.parallel (%arg0, %arg1) = (0, 0) to (32, 16) {
    %3 = memref.alloc() : memref<2x8x128xf32, 2>
    %4 = memref.alloc() : memref<2x8x128xf32, 2>
    %5 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
    %6 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg1)
    affine.parallel (%arg2, %arg3) = (0, 0) to (16, 16) {
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c0_0 = arith.constant 0 : index
      %c7 = arith.constant 7 : index
      %c0_1 = arith.constant 0 : index
      %c0_2 = arith.constant 0 : index
      %c0_3 = arith.constant 0 : index
      %c0_4 = arith.constant 0 : index
      %7 = memref.alloc() : memref<4xf32, 3>
      %8 = memref.alloc() : memref<4xf32, 3>
      %9 = memref.alloc() : memref<2x8xf32, 3>
      %10 = memref.alloc() : memref<2x8xf32, 3>
      %11 = memref.alloc() : memref<8x8xf32, 3>
      %12 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg2)
      %13 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg3)
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %cst = arith.constant 0.000000e+00 : f32
          affine.store %cst, %11[%arg4, %arg5] : memref<8x8xf32, 3>
        }
      }
      %14 = affine.vector_load %0[%arg2 * 8 + %arg3 floordiv 2 + %c0 * 128 + %arg0 * 128, (%arg3 mod 2) * 4 + %c0_3] : memref<4096x1024xf32, 1>, vector<4xf32>
      affine.vector_store %14, %7[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
      %15 = affine.vector_load %1[(%arg2 * 16 + %arg3) floordiv 32 + %c0 * 8 + %c0_4, ((%arg2 * 16 + %arg3) mod 32) * 4 + %arg1 * 128] : memref<1024x2048xf32, 1>, vector<4xf32>
      affine.vector_store %15, %8[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
      affine.for %arg4 = 0 to 4 {
        %21 = affine.vector_load %7[%c0 * 4 + %arg4] : memref<4xf32, 3>, vector<1xf32>
        affine.vector_store %21, %3[0, (%arg3 mod 2) * 4 + %arg4, %arg2 * 8 + %arg3 floordiv 2 + %c0 * 128] : memref<2x8x128xf32, 2>, vector<1xf32>
      }
      %16 = affine.vector_load %8[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
      affine.vector_store %16, %4[0, (%arg2 * 16 + %arg3) floordiv 32 + %c0 * 8, ((%arg2 * 16 + %arg3) mod 32) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
      gpu.barrier
      %17 = affine.vector_load %3[(%c0_0 floordiv 8) mod 2, %c0_1, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %c0 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
      affine.vector_store %17, %9[0, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
      %18 = affine.vector_load %3[(%c0_0 floordiv 8) mod 2, %c0_1, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %c1 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
      affine.vector_store %18, %9[0, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>
      %19 = affine.vector_load %4[(%c0 floordiv 8) mod 2, %c0_2, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %c0 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
      affine.vector_store %19, %10[0, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
      %20 = affine.vector_load %4[(%c0 floordiv 8) mod 2, %c0_2, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %c1 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
      affine.vector_store %20, %10[0, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>
      affine.for %arg4 = 0 to 1024 step 8 {
        affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg4) {
          %25 = affine.vector_load %0[%arg2 * 8 + %arg3 floordiv 2 + %c0 * 128 + %arg0 * 128, (%arg3 mod 2) * 4 + %arg4 + 8] : memref<4096x1024xf32, 1>, vector<4xf32>
          affine.vector_store %25, %7[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
          %26 = affine.vector_load %1[(%arg2 * 16 + %arg3) floordiv 32 + %c0 * 8 + %arg4 + 8, ((%arg2 * 16 + %arg3) mod 32) * 4 + %arg1 * 128] : memref<1024x2048xf32, 1>, vector<4xf32>
          affine.vector_store %26, %8[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
        }
        affine.for %arg5 = 0 to 7 {
          %25 = affine.vector_load %3[(%arg4 floordiv 8) mod 2, %arg5 + 1, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %c0 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %25, %9[(%arg5 + 1) mod 2, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
          %26 = affine.vector_load %3[(%arg4 floordiv 8) mod 2, %arg5 + 1, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %c1 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %26, %9[(%arg5 + 1) mod 2, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>
          %27 = affine.vector_load %4[(%arg4 floordiv 8) mod 2, %arg5 + 1, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %c0 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %27, %10[(%arg5 + 1) mod 2, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
          %28 = affine.vector_load %4[(%arg4 floordiv 8) mod 2, %arg5 + 1, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %c1 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %28, %10[(%arg5 + 1) mod 2, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>
          affine.for %arg6 = 0 to 8 {
            affine.for %arg7 = 0 to 8 {
              %29 = affine.load %11[%arg6, %arg7] : memref<8x8xf32, 3>
              %30 = affine.load %9[%arg5 mod 2, %arg6] : memref<2x8xf32, 3>
              %31 = affine.load %10[%arg5 mod 2, %arg7] : memref<2x8xf32, 3>
              %32 = arith.mulf %30, %31 : f32
              %33 = arith.addf %32, %29 : f32
              affine.store %33, %11[%arg6, %arg7] : memref<8x8xf32, 3>
            }
          }
        }
        affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg4) {
          affine.for %arg5 = 0 to 4 {
            %26 = affine.vector_load %7[%c0 * 4 + %arg5] : memref<4xf32, 3>, vector<1xf32>
            affine.vector_store %26, %3[(%arg4 floordiv 8 + 1) mod 2, (%arg3 mod 2) * 4 + %arg5, %arg2 * 8 + %arg3 floordiv 2 + %c0 * 128] : memref<2x8x128xf32, 2>, vector<1xf32>
          }
          %25 = affine.vector_load %8[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
          affine.vector_store %25, %4[(%arg4 floordiv 8 + 1) mod 2, (%arg2 * 16 + %arg3) floordiv 32 + %c0 * 8, ((%arg2 * 16 + %arg3) mod 32) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          gpu.barrier
        }
        affine.for %arg5 = 0 to 8 {
          affine.for %arg6 = 0 to 8 {
            %25 = affine.load %11[%arg5, %arg6] : memref<8x8xf32, 3>
            %26 = affine.load %9[%c7 mod 2, %arg5] : memref<2x8xf32, 3>
            %27 = affine.load %10[%c7 mod 2, %arg6] : memref<2x8xf32, 3>
            %28 = arith.mulf %26, %27 : f32
            %29 = arith.addf %28, %25 : f32
            affine.store %29, %11[%arg5, %arg6] : memref<8x8xf32, 3>
          }
        }
        %21 = affine.vector_load %4[(%arg4 floordiv 8) mod 2, %c0_2, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %c0 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
        affine.vector_store %21, %10[0, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
        %22 = affine.vector_load %4[(%arg4 floordiv 8) mod 2, %c0_2, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %c1 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
        affine.vector_store %22, %10[0, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>
        %23 = affine.vector_load %3[(%arg4 floordiv 8) mod 2, %c0_1, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %c0 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
        affine.vector_store %23, %9[0, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
        %24 = affine.vector_load %3[(%arg4 floordiv 8) mod 2, %c0_1, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %c1 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
        affine.vector_store %24, %9[0, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>
      }
      affine.for %arg4 = 0 to 4 {
        %21 = affine.vector_load %11[%c0 + %arg4, %c0 + %c0] : memref<8x8xf32, 3>, vector<4xf32>
        affine.vector_store %21, %2[%arg4 + %c0 * 16 + %arg2 * 4 + %arg0 * 128, %c0 + %c0 * 16 + %arg3 * 4 + %arg1 * 128] : memref<4096x2048xf32, 1>, vector<4xf32>
      }
      affine.for %arg4 = 0 to 4 {
        %21 = affine.vector_load %11[%c0 + %arg4, %c4 + %c0] : memref<8x8xf32, 3>, vector<4xf32>
        affine.vector_store %21, %2[%arg4 + %c0 * 16 + %arg2 * 4 + %arg0 * 128, %c0 + %c4 * 16 + %arg3 * 4 + %arg1 * 128] : memref<4096x2048xf32, 1>, vector<4xf32>
      }
      affine.for %arg4 = 0 to 4 {
        %21 = affine.vector_load %11[%c4 + %arg4, %c0 + %c0] : memref<8x8xf32, 3>, vector<4xf32>
        affine.vector_store %21, %2[%arg4 + %c4 * 16 + %arg2 * 4 + %arg0 * 128, %c0 + %c0 * 16 + %arg3 * 4 + %arg1 * 128] : memref<4096x2048xf32, 1>, vector<4xf32>
      }
      affine.for %arg4 = 0 to 4 {
        %21 = affine.vector_load %11[%c4 + %arg4, %c4 + %c0] : memref<8x8xf32, 3>, vector<4xf32>
        affine.vector_store %21, %2[%arg4 + %c4 * 16 + %arg2 * 4 + %arg0 * 128, %c0 + %c4 * 16 + %arg3 * 4 + %arg1 * 128] : memref<4096x2048xf32, 1>, vector<4xf32>
      }
    }
  }
}
```
loop unroll attribute
```c++
module @fuse_matmul_relu {
  %0 = memref.alloc() : memref<4096x1024xf32, 1>
  %1 = memref.alloc() : memref<1024x2048xf32, 1>
  %2 = memref.alloc() : memref<4096x2048xf32, 1>
  affine.parallel (%arg0, %arg1) = (0, 0) to (32, 16) {
    %3 = memref.alloc() : memref<2x8x128xf32, 2>
    %4 = memref.alloc() : memref<2x8x128xf32, 2>
    %5 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
    %6 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg1)
    affine.parallel (%arg2, %arg3) = (0, 0) to (16, 16) {
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c0_0 = arith.constant 0 : index
      %c7 = arith.constant 7 : index
      %c0_1 = arith.constant 0 : index
      %c0_2 = arith.constant 0 : index
      %c0_3 = arith.constant 0 : index
      %c0_4 = arith.constant 0 : index
      %7 = memref.alloc() : memref<4xf32, 3>
      %8 = memref.alloc() : memref<4xf32, 3>
      %9 = memref.alloc() : memref<2x8xf32, 3>
      %10 = memref.alloc() : memref<2x8xf32, 3>
      %11 = memref.alloc() : memref<8x8xf32, 3>
      %12 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg2)
      %13 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg3)
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %cst = arith.constant 0.000000e+00 : f32
          affine.store %cst, %11[%arg4, %arg5] : memref<8x8xf32, 3>
        } {affine.loop = "unroll"}
      } {affine.loop = "unroll"}
      %14 = affine.vector_load %0[%arg2 * 8 + %arg3 floordiv 2 + %c0 * 128 + %arg0 * 128, (%arg3 mod 2) * 4 + %c0_3] : memref<4096x1024xf32, 1>, vector<4xf32>
      affine.vector_store %14, %7[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
      %15 = affine.vector_load %1[(%arg2 * 16 + %arg3) floordiv 32 + %c0 * 8 + %c0_4, ((%arg2 * 16 + %arg3) mod 32) * 4 + %arg1 * 128] : memref<1024x2048xf32, 1>, vector<4xf32>
      affine.vector_store %15, %8[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
      affine.for %arg4 = 0 to 4 {
        %21 = affine.vector_load %7[%c0 * 4 + %arg4] : memref<4xf32, 3>, vector<1xf32>
        affine.vector_store %21, %3[0, (%arg3 mod 2) * 4 + %arg4, %arg2 * 8 + %arg3 floordiv 2 + %c0 * 128] : memref<2x8x128xf32, 2>, vector<1xf32>
      } {affine.loop = "unroll"}
      %16 = affine.vector_load %8[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
      affine.vector_store %16, %4[0, (%arg2 * 16 + %arg3) floordiv 32 + %c0 * 8, ((%arg2 * 16 + %arg3) mod 32) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
      gpu.barrier
      %17 = affine.vector_load %3[(%c0_0 floordiv 8) mod 2, %c0_1, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %c0 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
      affine.vector_store %17, %9[0, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
      %18 = affine.vector_load %3[(%c0_0 floordiv 8) mod 2, %c0_1, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %c1 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
      affine.vector_store %18, %9[0, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>
      %19 = affine.vector_load %4[(%c0 floordiv 8) mod 2, %c0_2, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %c0 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
      affine.vector_store %19, %10[0, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
      %20 = affine.vector_load %4[(%c0 floordiv 8) mod 2, %c0_2, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %c1 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
      affine.vector_store %20, %10[0, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>
      affine.for %arg4 = 0 to 1024 step 8 {
        affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg4) {
          %25 = affine.vector_load %0[%arg2 * 8 + %arg3 floordiv 2 + %c0 * 128 + %arg0 * 128, (%arg3 mod 2) * 4 + %arg4 + 8] : memref<4096x1024xf32, 1>, vector<4xf32>
          affine.vector_store %25, %7[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
          %26 = affine.vector_load %1[(%arg2 * 16 + %arg3) floordiv 32 + %c0 * 8 + %arg4 + 8, ((%arg2 * 16 + %arg3) mod 32) * 4 + %arg1 * 128] : memref<1024x2048xf32, 1>, vector<4xf32>
          affine.vector_store %26, %8[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
        }
        affine.for %arg5 = 0 to 7 {
          %25 = affine.vector_load %3[(%arg4 floordiv 8) mod 2, %arg5 + 1, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %c0 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %25, %9[(%arg5 + 1) mod 2, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
          %26 = affine.vector_load %3[(%arg4 floordiv 8) mod 2, %arg5 + 1, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %c1 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %26, %9[(%arg5 + 1) mod 2, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>
          %27 = affine.vector_load %4[(%arg4 floordiv 8) mod 2, %arg5 + 1, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %c0 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %27, %10[(%arg5 + 1) mod 2, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
          %28 = affine.vector_load %4[(%arg4 floordiv 8) mod 2, %arg5 + 1, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %c1 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %28, %10[(%arg5 + 1) mod 2, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>
          affine.for %arg6 = 0 to 8 {
            affine.for %arg7 = 0 to 8 {
              %29 = affine.load %11[%arg6, %arg7] : memref<8x8xf32, 3>
              %30 = affine.load %9[%arg5 mod 2, %arg6] : memref<2x8xf32, 3>
              %31 = affine.load %10[%arg5 mod 2, %arg7] : memref<2x8xf32, 3>
              %32 = arith.mulf %30, %31 : f32
              %33 = arith.addf %32, %29 : f32
              affine.store %33, %11[%arg6, %arg7] : memref<8x8xf32, 3>
            } {affine.loop = "unroll"}
          } {affine.loop = "unroll"}
        } {affine.loop = "unroll"}
        affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg4) {
          affine.for %arg5 = 0 to 4 {
            %26 = affine.vector_load %7[%c0 * 4 + %arg5] : memref<4xf32, 3>, vector<1xf32>
            affine.vector_store %26, %3[(%arg4 floordiv 8 + 1) mod 2, (%arg3 mod 2) * 4 + %arg5, %arg2 * 8 + %arg3 floordiv 2 + %c0 * 128] : memref<2x8x128xf32, 2>, vector<1xf32>
          } {affine.loop = "unroll"}
          %25 = affine.vector_load %8[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
          affine.vector_store %25, %4[(%arg4 floordiv 8 + 1) mod 2, (%arg2 * 16 + %arg3) floordiv 32 + %c0 * 8, ((%arg2 * 16 + %arg3) mod 32) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          gpu.barrier
        }
        affine.for %arg5 = 0 to 8 {
          affine.for %arg6 = 0 to 8 {
            %25 = affine.load %11[%arg5, %arg6] : memref<8x8xf32, 3>
            %26 = affine.load %9[%c7 mod 2, %arg5] : memref<2x8xf32, 3>
            %27 = affine.load %10[%c7 mod 2, %arg6] : memref<2x8xf32, 3>
            %28 = arith.mulf %26, %27 : f32
            %29 = arith.addf %28, %25 : f32
            affine.store %29, %11[%arg5, %arg6] : memref<8x8xf32, 3>
          } {affine.loop = "unroll"}
        } {affine.loop = "unroll"}
        %21 = affine.vector_load %4[(%arg4 floordiv 8) mod 2, %c0_2, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %c0 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
        affine.vector_store %21, %10[0, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
        %22 = affine.vector_load %4[(%arg4 floordiv 8) mod 2, %c0_2, (%arg3 mod 4 + (((%arg2 * 16 + %arg3) floordiv 32) mod 4 + %c1 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
        affine.vector_store %22, %10[0, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>
        %23 = affine.vector_load %3[(%arg4 floordiv 8) mod 2, %c0_1, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %c0 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
        affine.vector_store %23, %9[0, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
        %24 = affine.vector_load %3[(%arg4 floordiv 8) mod 2, %c0_1, (((%arg2 * 16 + %arg3) mod 32) floordiv 4 + (((%arg2 * 16 + %arg3) floordiv 32) floordiv 4 + %c1 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
        affine.vector_store %24, %9[0, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>
      }
      affine.for %arg4 = 0 to 4 {
        %21 = affine.vector_load %11[%c0 + %arg4, %c0 + %c0] : memref<8x8xf32, 3>, vector<4xf32>
        affine.vector_store %21, %2[%arg4 + %c0 * 16 + %arg2 * 4 + %arg0 * 128, %c0 + %c0 * 16 + %arg3 * 4 + %arg1 * 128] : memref<4096x2048xf32, 1>, vector<4xf32>
      } {affine.loop = "unroll"}
      affine.for %arg4 = 0 to 4 {
        %21 = affine.vector_load %11[%c0 + %arg4, %c4 + %c0] : memref<8x8xf32, 3>, vector<4xf32>
        affine.vector_store %21, %2[%arg4 + %c0 * 16 + %arg2 * 4 + %arg0 * 128, %c0 + %c4 * 16 + %arg3 * 4 + %arg1 * 128] : memref<4096x2048xf32, 1>, vector<4xf32>
      } {affine.loop = "unroll"}
      affine.for %arg4 = 0 to 4 {
        %21 = affine.vector_load %11[%c4 + %arg4, %c0 + %c0] : memref<8x8xf32, 3>, vector<4xf32>
        affine.vector_store %21, %2[%arg4 + %c4 * 16 + %arg2 * 4 + %arg0 * 128, %c0 + %c0 * 16 + %arg3 * 4 + %arg1 * 128] : memref<4096x2048xf32, 1>, vector<4xf32>
      } {affine.loop = "unroll"}
      affine.for %arg4 = 0 to 4 {
        %21 = affine.vector_load %11[%c4 + %arg4, %c4 + %c0] : memref<8x8xf32, 3>, vector<4xf32>
        affine.vector_store %21, %2[%arg4 + %c4 * 16 + %arg2 * 4 + %arg0 * 128, %c0 + %c4 * 16 + %arg3 * 4 + %arg1 * 128] : memref<4096x2048xf32, 1>, vector<4xf32>
      } {affine.loop = "unroll"}
    }
  }
}
```