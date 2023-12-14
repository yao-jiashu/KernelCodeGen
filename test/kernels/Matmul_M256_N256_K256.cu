#include "cuda_runtime.h"
namespace Matmul_M256_N256_K256 {
// grid dims:(2, 2, ), block dims:(16, 16, )
__global__ void kernel0(float* arg0, float* arg1, float* arg2) {
  __shared__ float array0[2][8][128];
  __shared__ float array1[2][8][128];
  int expr0 = (blockIdx.y * 128);
  int expr1 = (blockIdx.x * 128);
  constexpr int const0th = 4;
  constexpr int const1th = 1;
  constexpr int const2th = 0;
  constexpr int const3th = 0;
  constexpr int const4th = 7;
  constexpr int const5th = 0;
  constexpr int const6th = 0;
  constexpr int const7th = 0;
  constexpr int const8th = 0;
  float array2[4];
  float array3[4];
  float array4[2][8];
  float array5[2][8];
  float array6[8][8];
  int expr2 = (threadIdx.y * 8);
  int expr3 = (threadIdx.x * 8);
  #pragma unroll
  for (int iter0 = 0; iter0 < 8; iter0 += 1) {
    #pragma unroll
    for (int iter1 = 0; iter1 < 8; iter1 += 1) {
      constexpr float const9th = 0;
      array6[iter0][iter1] = const9th;
    }
  }
  auto vec0 = (reinterpret_cast<float4*>(&(arg0[((((threadIdx.y * 8) + (threadIdx.x / 2)) + (const2th * 128)) + (blockIdx.y * 128)) * 256 + (((threadIdx.x % 2) * 4) + const7th) * 1 + 0]))[0]);
  (reinterpret_cast<float4*>(&(array2[(const2th * 4)]))[0]) = vec0;
  auto vec1 = (reinterpret_cast<float4*>(&(arg1[(((((threadIdx.y * 16) + threadIdx.x) / 32) + (const2th * 8)) + const8th) * 256 + (((((threadIdx.y * 16) + threadIdx.x) % 32) * 4) + (blockIdx.x * 128)) * 1 + 0]))[0]);
  (reinterpret_cast<float4*>(&(array3[(const2th * 4)]))[0]) = vec1;
  #pragma unroll
  for (int iter2 = 0; iter2 < 4; iter2 += 1) {
    auto vec2 = (reinterpret_cast<float1*>(&(array2[((const2th * 4) + iter2)]))[0]);
    (reinterpret_cast<float1*>(&(array0[0][(((threadIdx.x % 2) * 4) + iter2)][(((threadIdx.y * 8) + (threadIdx.x / 2)) + (const2th * 128))]))[0]) = vec2;
  }
  auto vec3 = (reinterpret_cast<float4*>(&(array3[(const2th * 4)]))[0]);
  (reinterpret_cast<float4*>(&(array1[0][((((threadIdx.y * 16) + threadIdx.x) / 32) + (const2th * 8))][((((threadIdx.y * 16) + threadIdx.x) % 32) * 4)]))[0]) = vec3;
  __syncthreads();
  auto vec4 = (reinterpret_cast<float4*>(&(array0[((const3th / 8) % 2)][const5th][((((((threadIdx.y * 16) + threadIdx.x) % 32) / 4) + ((((((threadIdx.y * 16) + threadIdx.x) / 32) / 4) + (const2th * 2)) * 8)) * 4)]))[0]);
  (reinterpret_cast<float4*>(&(array4[0][(const2th * 4)]))[0]) = vec4;
  auto vec5 = (reinterpret_cast<float4*>(&(array0[((const3th / 8) % 2)][const5th][((((((threadIdx.y * 16) + threadIdx.x) % 32) / 4) + ((((((threadIdx.y * 16) + threadIdx.x) / 32) / 4) + (const1th * 2)) * 8)) * 4)]))[0]);
  (reinterpret_cast<float4*>(&(array4[0][(const1th * 4)]))[0]) = vec5;
  auto vec6 = (reinterpret_cast<float4*>(&(array1[((const2th / 8) % 2)][const6th][(((threadIdx.x % 4) + ((((((threadIdx.y * 16) + threadIdx.x) / 32) % 4) + (const2th * 4)) * 4)) * 4)]))[0]);
  (reinterpret_cast<float4*>(&(array5[0][(const2th * 4)]))[0]) = vec6;
  auto vec7 = (reinterpret_cast<float4*>(&(array1[((const2th / 8) % 2)][const6th][(((threadIdx.x % 4) + ((((((threadIdx.y * 16) + threadIdx.x) / 32) % 4) + (const1th * 4)) * 4)) * 4)]))[0]);
  (reinterpret_cast<float4*>(&(array5[0][(const1th * 4)]))[0]) = vec7;
  for (int iter3 = 0; iter3 < 256; iter3 += 8) {
    if (((iter3 * -1) + 240) >= 0 &&  true) {
      auto vec8 = (reinterpret_cast<float4*>(&(arg0[((((threadIdx.y * 8) + (threadIdx.x / 2)) + (const2th * 128)) + (blockIdx.y * 128)) * 256 + (((threadIdx.x % 2) * 4) + (iter3 + 8)) * 1 + 0]))[0]);
      (reinterpret_cast<float4*>(&(array2[(const2th * 4)]))[0]) = vec8;
      auto vec9 = (reinterpret_cast<float4*>(&(arg1[(((((threadIdx.y * 16) + threadIdx.x) / 32) + (const2th * 8)) + (iter3 + 8)) * 256 + (((((threadIdx.y * 16) + threadIdx.x) % 32) * 4) + (blockIdx.x * 128)) * 1 + 0]))[0]);
      (reinterpret_cast<float4*>(&(array3[(const2th * 4)]))[0]) = vec9;
    }
    #pragma unroll
    for (int iter4 = 0; iter4 < 7; iter4 += 1) {
      auto vec10 = (reinterpret_cast<float4*>(&(array0[((iter3 / 8) % 2)][(iter4 + 1)][((((((threadIdx.y * 16) + threadIdx.x) % 32) / 4) + ((((((threadIdx.y * 16) + threadIdx.x) / 32) / 4) + (const2th * 2)) * 8)) * 4)]))[0]);
      (reinterpret_cast<float4*>(&(array4[((iter4 + 1) % 2)][(const2th * 4)]))[0]) = vec10;
      auto vec11 = (reinterpret_cast<float4*>(&(array0[((iter3 / 8) % 2)][(iter4 + 1)][((((((threadIdx.y * 16) + threadIdx.x) % 32) / 4) + ((((((threadIdx.y * 16) + threadIdx.x) / 32) / 4) + (const1th * 2)) * 8)) * 4)]))[0]);
      (reinterpret_cast<float4*>(&(array4[((iter4 + 1) % 2)][(const1th * 4)]))[0]) = vec11;
      auto vec12 = (reinterpret_cast<float4*>(&(array1[((iter3 / 8) % 2)][(iter4 + 1)][(((threadIdx.x % 4) + ((((((threadIdx.y * 16) + threadIdx.x) / 32) % 4) + (const2th * 4)) * 4)) * 4)]))[0]);
      (reinterpret_cast<float4*>(&(array5[((iter4 + 1) % 2)][(const2th * 4)]))[0]) = vec12;
      auto vec13 = (reinterpret_cast<float4*>(&(array1[((iter3 / 8) % 2)][(iter4 + 1)][(((threadIdx.x % 4) + ((((((threadIdx.y * 16) + threadIdx.x) / 32) % 4) + (const1th * 4)) * 4)) * 4)]))[0]);
      (reinterpret_cast<float4*>(&(array5[((iter4 + 1) % 2)][(const1th * 4)]))[0]) = vec13;
      #pragma unroll
      for (int iter5 = 0; iter5 < 8; iter5 += 1) {
        #pragma unroll
        for (int iter6 = 0; iter6 < 8; iter6 += 1) {
          auto R0 = array6[iter5][iter6];
          auto R1 = array4[(iter4 % 2)][iter5];
          auto R2 = array5[(iter4 % 2)][iter6];
          auto temp0 = R1 * R2;
          auto temp2 = temp0 + R0;
          array6[iter5][iter6] = temp2;
        }
      }
    }
    if (((iter3 * -1) + 240) >= 0 &&  true) {
      #pragma unroll
      for (int iter7 = 0; iter7 < 4; iter7 += 1) {
        auto vec14 = (reinterpret_cast<float1*>(&(array2[((const2th * 4) + iter7)]))[0]);
        (reinterpret_cast<float1*>(&(array0[(((iter3 / 8) + 1) % 2)][(((threadIdx.x % 2) * 4) + iter7)][(((threadIdx.y * 8) + (threadIdx.x / 2)) + (const2th * 128))]))[0]) = vec14;
      }
      auto vec15 = (reinterpret_cast<float4*>(&(array3[(const2th * 4)]))[0]);
      (reinterpret_cast<float4*>(&(array1[(((iter3 / 8) + 1) % 2)][((((threadIdx.y * 16) + threadIdx.x) / 32) + (const2th * 8))][((((threadIdx.y * 16) + threadIdx.x) % 32) * 4)]))[0]) = vec15;
      __syncthreads();
    }
    #pragma unroll
    for (int iter8 = 0; iter8 < 8; iter8 += 1) {
      #pragma unroll
      for (int iter9 = 0; iter9 < 8; iter9 += 1) {
        auto R3 = array6[iter8][iter9];
        auto R4 = array4[(const4th % 2)][iter8];
        auto R5 = array5[(const4th % 2)][iter9];
        auto temp1 = R4 * R5;
        auto temp3 = temp1 + R3;
        array6[iter8][iter9] = temp3;
      }
    }
    auto vec16 = (reinterpret_cast<float4*>(&(array1[(((iter3 / 8) + 1) % 2)][const6th][(((threadIdx.x % 4) + ((((((threadIdx.y * 16) + threadIdx.x) / 32) % 4) + (const2th * 4)) * 4)) * 4)]))[0]);
    (reinterpret_cast<float4*>(&(array5[0][(const2th * 4)]))[0]) = vec16;
    auto vec17 = (reinterpret_cast<float4*>(&(array1[(((iter3 / 8) + 1) % 2)][const6th][(((threadIdx.x % 4) + ((((((threadIdx.y * 16) + threadIdx.x) / 32) % 4) + (const1th * 4)) * 4)) * 4)]))[0]);
    (reinterpret_cast<float4*>(&(array5[0][(const1th * 4)]))[0]) = vec17;
    auto vec18 = (reinterpret_cast<float4*>(&(array0[(((iter3 / 8) + 1) % 2)][const5th][((((((threadIdx.y * 16) + threadIdx.x) % 32) / 4) + ((((((threadIdx.y * 16) + threadIdx.x) / 32) / 4) + (const2th * 2)) * 8)) * 4)]))[0]);
    (reinterpret_cast<float4*>(&(array4[0][(const2th * 4)]))[0]) = vec18;
    auto vec19 = (reinterpret_cast<float4*>(&(array0[(((iter3 / 8) + 1) % 2)][const5th][((((((threadIdx.y * 16) + threadIdx.x) % 32) / 4) + ((((((threadIdx.y * 16) + threadIdx.x) / 32) / 4) + (const1th * 2)) * 8)) * 4)]))[0]);
    (reinterpret_cast<float4*>(&(array4[0][(const1th * 4)]))[0]) = vec19;
  }
  #pragma unroll
  for (int iter10 = 0; iter10 < 4; iter10 += 1) {
    auto vec20 = (reinterpret_cast<float4*>(&(array6[(const2th + iter10)][(const2th + const2th)]))[0]);
    (reinterpret_cast<float4*>(&(arg2[(((blockIdx.y * 128) + ((((((threadIdx.y * 16) + threadIdx.x) % 32) / 4) + ((((((threadIdx.y * 16) + threadIdx.x) / 32) / 4) + ((const2th / 4) * 2)) * 8)) * 4)) + iter10) * 256 + (((blockIdx.x * 128) + (((threadIdx.x % 4) + ((((((threadIdx.y * 16) + threadIdx.x) / 32) % 4) + ((const2th / 4) * 4)) * 4)) * 4)) + const2th) * 1 + 0]))[0]) = vec20;
  }
  #pragma unroll
  for (int iter11 = 0; iter11 < 4; iter11 += 1) {
    auto vec21 = (reinterpret_cast<float4*>(&(array6[(const2th + iter11)][(const0th + const2th)]))[0]);
    (reinterpret_cast<float4*>(&(arg2[(((blockIdx.y * 128) + ((((((threadIdx.y * 16) + threadIdx.x) % 32) / 4) + ((((((threadIdx.y * 16) + threadIdx.x) / 32) / 4) + ((const2th / 4) * 2)) * 8)) * 4)) + iter11) * 256 + (((blockIdx.x * 128) + (((threadIdx.x % 4) + ((((((threadIdx.y * 16) + threadIdx.x) / 32) % 4) + ((const0th / 4) * 4)) * 4)) * 4)) + const2th) * 1 + 0]))[0]) = vec21;
  }
  #pragma unroll
  for (int iter12 = 0; iter12 < 4; iter12 += 1) {
    auto vec22 = (reinterpret_cast<float4*>(&(array6[(const0th + iter12)][(const2th + const2th)]))[0]);
    (reinterpret_cast<float4*>(&(arg2[(((blockIdx.y * 128) + ((((((threadIdx.y * 16) + threadIdx.x) % 32) / 4) + ((((((threadIdx.y * 16) + threadIdx.x) / 32) / 4) + ((const0th / 4) * 2)) * 8)) * 4)) + iter12) * 256 + (((blockIdx.x * 128) + (((threadIdx.x % 4) + ((((((threadIdx.y * 16) + threadIdx.x) / 32) % 4) + ((const2th / 4) * 4)) * 4)) * 4)) + const2th) * 1 + 0]))[0]) = vec22;
  }
  #pragma unroll
  for (int iter13 = 0; iter13 < 4; iter13 += 1) {
    auto vec23 = (reinterpret_cast<float4*>(&(array6[(const0th + iter13)][(const0th + const2th)]))[0]);
    (reinterpret_cast<float4*>(&(arg2[(((blockIdx.y * 128) + ((((((threadIdx.y * 16) + threadIdx.x) % 32) / 4) + ((((((threadIdx.y * 16) + threadIdx.x) / 32) / 4) + ((const0th / 4) * 2)) * 8)) * 4)) + iter13) * 256 + (((blockIdx.x * 128) + (((threadIdx.x % 4) + ((((((threadIdx.y * 16) + threadIdx.x) / 32) % 4) + ((const0th / 4) * 4)) * 4)) * 4)) + const2th) * 1 + 0]))[0]) = vec23;
  }
}
}
