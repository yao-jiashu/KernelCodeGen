///usr/local/cuda-12.1/bin/nvcc matmul.cu  -arch=sm_80
#include <iostream>
#include <cstdlib>
#include <functional>
#include <cassert>
#include <unistd.h>
#include <type_traits>
#include <cublas_v2.h>
#include <curand.h>
#include <sstream>

#include "adaptor.cu"



// error check
#define CUDA_CHECK(call) do {                                                   \
  cudaError_t status = (call);                                                  \
  std::stringstream serr;                                                       \
  if (status != cudaSuccess) {                                                  \
    serr << "Cuda Error: " << status << " : " << cudaGetErrorString(status);    \
    serr << "\n" << __FILE__ << " : " << __LINE__;                              \
    std::cerr << serr.str() <<  "\nAborting...\n";                              \
    exit(1);                                                                    \
  }                                                                             \
} while(0)

// occupancy calculator
#define Occupancy(kernel, blockSize, dynamicSMemSize, device)                                         \
do {                                                                                                  \
  cudaGetDevice(&device);                                                                             \
  cudaDeviceProp prop;                                                                                \
  cudaGetDeviceProperties(&prop, (device));                                                           \
  int numberBlocks;                                                                                   \
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(                                                      \
      &numberBlocks,                                                                                  \
      (kernel),                                                                                       \
      (blockSize),                                                                                    \
      (dynamicSMemSize));                                                                             \
  int activeWarps = numberBlocks * (blockSize) / prop.warpSize;                                       \
  int maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;                                    \
  std::cout << "Occupancy: " << static_cast<double>(activeWarps) / maxWarps * 100 << "%" << std::endl;\
} while(0)


inline void printDeviceProp(const cudaDeviceProp &prop) {
  printf("[ Device Name : %s. ]\n", prop.name);
  printf("[ Compute Capability : %d.%d ]\n", prop.major, prop.minor);
  printf("[ Total Global Memory : %ld GB ]\n", prop.totalGlobalMem/ (1024 * 1024 * 1024) );
  printf("[ Total Const Memory : %ld KB ]\n", prop.totalConstMem/ 1024 );
  printf("[ SM counts : %d. ]\n", prop.multiProcessorCount);
  printf("[ Max Block counts Per SM : %d. ]\n", prop.maxBlocksPerMultiProcessor);
  printf("[ Max GridSize[0 - 2] : %d %d %d. ]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("[ Max ThreadsDim[0 - 2] : %d %d %d. ]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("[ Max Shared Memory Per SM : %ld KB ]\n", prop.sharedMemPerMultiprocessor/1024);
  printf("[ Max Shared Memory Per Block : %ld KB ]\n", prop.sharedMemPerBlock/1024);
  printf("[ Max Registers Number Per SM : %d. ]\n", prop.regsPerMultiprocessor);
  printf("[ Max Registers Number Per Block : %d. ]\n", prop.regsPerBlock);
  printf("[ Max Threads Number Per SM : %d. ]\n", prop.maxThreadsPerMultiProcessor);
  printf("[ Max Threads Number Per Block : %d. ]\n", prop.maxThreadsPerBlock);

  printf("[ warpSize : %d. ]\n", prop.warpSize);
  printf("[ memPitch : %ld. ]\n", prop.memPitch);
  printf("[ clockRate : %d. ]\n", prop.clockRate);
  printf("[ textureAlignment : %ld. ]\n", prop.textureAlignment);
  printf("[ deviceOverlap : %d. ]\n\n", prop.deviceOverlap);
}

/*Obtain computing device information and initialize the computing device*/
inline bool init_cuda(int verbose, int deviceId = 0) {
  int count;
  cudaGetDeviceCount(&count);
  if (count == 0) {
    std::cerr << "There is no device.\n";
    return false;
  }
  else {
    std::cout << "Find the device successfully.\n";
  }
  if (deviceId >= count) {
    std::cerr << "Device ID invalid";
    return false;
  }
  //set its value between 0 and n - 1 if there are n GPUS
  cudaSetDevice(deviceId);
  if (verbose == 1) {
    cudaDeviceProp prop {} ;
    cudaGetDeviceProperties(&prop, count -1);
    printDeviceProp(prop);
  }
  return true;
}

#define CURAND_CALL(x)                              \
do {                                                \
  if((x)!=CURAND_STATUS_SUCCESS) {                  \
    printf("Error at %s:%d\n",__FILE__,__LINE__);   \
    return EXIT_FAILURE;                            \
  }                                                 \
} while(0)

#define CUBLAS_CHECK(call) do {                                                     \
  cublasStatus_t status = (call);                                                   \
  std::stringstream serr;                                                           \
  if (status != CUBLAS_STATUS_SUCCESS) {                                            \
    serr << "CUBLAS Error: " << status << " : " << cublasGetStatusString(status);   \
    serr << "\n" << __FILE__ << " : " << __LINE__;                                  \
    std::cerr << serr.str() <<  "\nAborting...\n";                                  \
    exit(1);                                                                        \
  }                                                                                 \
} while(0)

int device = 0;
enum class Mode {
    Debug = 0,
    Perf = 1,
    All = 2,
};

// 主机端sleep几秒钟，库跑的就慢一些, 因为kernel之间可以共享cache，造成测量误差
#define REFRESH_CACHE sleep(8)

// -------------------------- static parameters configuration -----------------------------------//
const int BLOCK_SIZE_N = 128;
const int BLOCK_SIZE_M = 128;
const int BLOCK_SIZE_K = 8;
const int THREAD_SIZE_X = 8;
const int THREAD_SIZE_Y = 8;
dim3 dim_g = {N / BLOCK_SIZE_N, M / BLOCK_SIZE_M};
dim3 dim_b = {BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y};


// -------------------------- Select test performance or debug -----------------------------------//
//#define DEBUG
#define PERFORMANCE
#ifdef DEBUG
Mode testMode = Mode::Debug;
#elif defined(PERFORMANCE)
Mode testMode = Mode::Perf;
#else
Mode testMode = Mode::All;
#endif

const int warmup = 5;
const uint32_t n_trials = 10;
dim3 defaultBlockDim = dim3(16, 16);
dim3 defaultGrimDim = dim3((N + defaultBlockDim.x - 1) / defaultBlockDim.x,
                           (M + defaultBlockDim.y - 1) / defaultBlockDim.y);

const float epsilon = 1.0e-5;

template<typename DType>
void initHostMemory(DType *&A, DType *&B, DType *&C, uint32_t M, uint32_t N, uint32_t K) {
  srand(time(0));
  A = static_cast<DType *>(malloc(M * K * sizeof(DType)));
  B = static_cast<DType *>(malloc(K * N * sizeof(DType)));
  C = static_cast<DType *>(malloc(M * N * sizeof(DType)));
  #pragma omp parallel for
  for (uint32_t i = 0; i < M; i++)
    for (uint32_t j = 0; j < K; j++)
      A[i * K + j] = rand() % 10; // rand()返回0到一个整数之间的随机整数
  #pragma omp parallel for
  for (uint32_t i = 0; i < K; i++)
    for (uint32_t j = 0; j < N; j++)
      B[i * N + j] = rand() % 10;
  #pragma omp parallel for
  for (uint32_t i = 0; i < M; i++)
    for (uint32_t j = 0; j < N; j++)
      C[i * N + j] = rand() % 10;
}

template<typename DType>
void initDeviceMemory(DType *&d_A, DType *&d_B, DType *&d_C, DType *&verify_C,
                      DType *h_A, DType *h_B, DType *h_C, uint32_t M, uint32_t N, uint32_t K) {
  CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(DType)));
  CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(DType)));
  CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(DType)));
  CUDA_CHECK(cudaMalloc(&verify_C, M * N * sizeof(DType)));
  CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(DType), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(DType), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(DType), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(verify_C, h_C, M * N * sizeof(DType), cudaMemcpyHostToDevice));
}

template<class DType>
__global__ void verifyGemmTrans(const DType *C1, const DType *C2, uint32_t M, uint32_t N) {
  uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M and col < N) {
    if (std::abs(C1[row * N + col] - C2[col * M + row]) > epsilon) {
      printf("Row - %d, Col - %d -- Distance = %f\n", row, col, C1[row * N + col] - C2[col * M + row]);
      __trap();
    }
  }
}

template<class DType>
__global__ void verifyGemm(const DType *C1, const DType *C2, uint32_t M, uint32_t N) {
  uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M and col < N) {
    if (std::abs(C1[row * N + col] - C2[row * N + col]) > epsilon) {
      printf("Row - %d, Col - %d -- Distance = %f\n", row, col, C1[row * N + col] - C2[row * N + col]);
      __trap();
    }
  }
}

template<typename DType>
void testMatmul(dim3 dimGrid, dim3 dimBlock, uint32_t M, uint32_t N, uint32_t K) {
  DType *h_A{nullptr}, *h_B{nullptr}, *h_C{nullptr}, *d_A{nullptr}, *d_B{nullptr}, *d_C{nullptr}, *verify_C{nullptr};
  initHostMemory<DType>(h_A, h_B, h_C, M, N, K);
  initDeviceMemory<DType>(d_A, d_B, d_C, verify_C, h_A, h_B, h_C,  M, N, K);

  if (testMode == Mode::All or testMode == Mode::Debug) {
    //kernel实现和cublas的beta参数（设为0）都忽略了初始值
    verifyGemm<DType><<<defaultGrimDim, defaultBlockDim>>>(verify_C, d_C, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    std::cout << "Init  = Pass\n";

    // kernel correctness
    std::cout << "grid size = " << dimGrid.y << ", " << dimGrid.x << "\n";
    std::cout << "block size = " << dimBlock.y << ", " << dimBlock.x << "\n";
    // matmul_04<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_SIZE_Y, THREAD_SIZE_X><<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    kernelFunc<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    // kernelFunc<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    {
      cublasHandle_t handle;
      // 考虑 A×B = C，则 B^T × A^T = C^T
      // 我们已有的数据就是  B^T(N * K, N这个维度先存) A^T(K * M, K这个维度先存),得到的结果是C^T(N * M, N这个维度先存),从结果来看还是行优先的
      cublasOperation_t transB = CUBLAS_OP_N, transA = CUBLAS_OP_N;
      int ldb = N, lda = K, ldc = N;
      const float alpha{1.0}, beta{0.0};
      CUBLAS_CHECK(cublasCreate(&handle));
      CUBLAS_CHECK(cublasSgemm(handle, transB, transA,
                              N, M, K, &alpha, d_B, ldb,
                              d_A, lda, &beta, verify_C, ldc));
      verifyGemm<DType><<<dimGrid, dimBlock>>>(verify_C, d_C, M, N);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaGetLastError());
      std::cout << "Compare Correctness to cublas(NN) = Pass" << std::endl;
    }
    {
      // cublasHandle_t handle;
      // cublasOperation_t transA = CUBLAS_OP_T, transB = CUBLAS_OP_T;
      // // 因为要转置,需要给出转置之前按哪个维度存的,但是结果是列优先存储的,所以需要给出C的行数
      // int lda = K, ldb = N, ldc = M;
      // const float alpha{1.0}, beta{0.0};
      // CUBLAS_CHECK(cublasCreate(&handle));
      // CUBLAS_CHECK(cublasSgemm(handle, transA, transB,
      //                         M, N, K, &alpha, d_A, lda,
      //                         d_B, ldb, &beta, verify_C, ldc));
      // verifyGemmTrans<DType><<<dimGrid, dimBlock>>>(d_C, verify_C, M, N);
      // CUDA_CHECK(cudaDeviceSynchronize());
      // CUDA_CHECK(cudaGetLastError());
      // std::cout << "Compare Correctness to cublas(TT) = Pass" << std::endl;
    }
  }
  std::cout << std::endl;
  if (testMode == Mode::All or testMode == Mode::Perf) {
    float cublasTTime, cublasNTime, myGemmTime;
    {
      REFRESH_CACHE;
      cublasHandle_t handle;
      cublasOperation_t transA = CUBLAS_OP_N, transB = CUBLAS_OP_N;
      int lda = K, ldb = N, ldc = N;
      const float alpha{1.0}, beta{0.0};
      CUBLAS_CHECK(cublasCreate(&handle));
      for (int32_t i = 0; i < warmup; i++) {
        CUBLAS_CHECK(cublasSgemm(handle, transB, transA,
                                N, M, K, &alpha, d_B, ldb,
                                d_A, lda, &beta, verify_C, ldc));
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaGetLastError());
  
      cudaEvent_t start, stop;
      CUDA_CHECK(cudaEventCreate(&start));
      CUDA_CHECK(cudaEventCreate(&stop));
      CUDA_CHECK(cudaEventRecord(start, 0));
  
      for (int32_t i = 0; i < n_trials; i++) {
        CUBLAS_CHECK(cublasSgemm(handle, transB, transA,
                                N, M, K, &alpha, d_B, ldb,
                                d_A, lda, &beta, verify_C, ldc));
      }
      CUDA_CHECK(cudaEventRecord(stop, 0));
      CUDA_CHECK(cudaEventSynchronize(stop));
      cudaEventElapsedTime(&cublasNTime, start, stop);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
    }
    {
      REFRESH_CACHE;
      cublasHandle_t handle;
      cublasOperation_t transA = CUBLAS_OP_T, transB = CUBLAS_OP_T;
      int lda = K, ldb = N, ldc = M;
      const float alpha{1.0}, beta{0.0};
      CUBLAS_CHECK(cublasCreate(&handle));
      for (int32_t i = 0; i < warmup; i++) {
        CUBLAS_CHECK(cublasSgemm(handle, transA, transB,
                                M, N, K, &alpha, d_A, lda,
                                d_B, ldb, &beta, verify_C, ldc));
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaGetLastError());
  
      cudaEvent_t start, stop;
      CUDA_CHECK(cudaEventCreate(&start));
      CUDA_CHECK(cudaEventCreate(&stop));
      CUDA_CHECK(cudaEventRecord(start, 0));
  
      for (int32_t i = 0; i < n_trials; i++) {
        CUBLAS_CHECK(cublasSgemm(handle, transA, transB,
                                M, N, K, &alpha, d_A, lda,
                                d_B, ldb, &beta, verify_C, ldc));
      }
      CUDA_CHECK(cudaEventRecord(stop, 0));
      CUDA_CHECK(cudaEventSynchronize(stop));
      cudaEventElapsedTime(&cublasTTime, start, stop);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
    }
    {
      REFRESH_CACHE;
      // warmup
      for (int32_t i = 0; i < warmup; i++) {
        kernelFunc<<< dimGrid, dimBlock >>>(d_A, d_B, d_C);
        // kernelFunc<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaGetLastError());

      cudaEvent_t start, stop;
      CUDA_CHECK(cudaEventCreate(&start));
      CUDA_CHECK(cudaEventCreate(&stop));
      CUDA_CHECK(cudaEventRecord(start, 0));

      for (int32_t i = 0; i < n_trials; i++) {
        kernelFunc<<<dimGrid, dimBlock >>>(d_A, d_B, d_C);
        // kernelFunc<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
        // 放在循环内，反而会因为和CPU同步影响结果，而且同一个流中的 kernel函数是串行的
        // cudaDeviceSynchronize();
      }
      CUDA_CHECK(cudaEventRecord(stop, 0));
      CUDA_CHECK(cudaEventSynchronize(stop));
      cudaEventElapsedTime(&myGemmTime, start, stop);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
    }

    Occupancy(kernelFunc, dimBlock.x * dimBlock.y, 0, device);
    std::cout << "My Matmul Latency = " << myGemmTime / n_trials << " ms\n\n";

    std::cout << "cublas Latency(NN) = " << cublasNTime / n_trials << " ms\n";
    std::cout << "cublas Ratio(NN) = " << cublasNTime / myGemmTime * 100  << "%\n" << std::endl;

    std::cout << "cublas Latency(TT) = " << cublasTTime / n_trials << " ms\n";
    std::cout << "cublas Ratio(TT) = " << cublasTTime / myGemmTime * 100  << "%\n" << std::endl;
  }
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(verify_C);
  free(h_A);
  free(h_B);
  free(h_C);
}

int main(int argc, char *argv[]) {
  init_cuda(0, device);
  cudaFree(0);
  std::cout << "Matrix size : " << std::endl
    << " --- M --- : " << M << std::endl
    << " --- N --- : " << N << std::endl
    << " --- K --- : " << K << std::endl;
  testMatmul<float>(dim_g, dim_b, M, N, K);
  return 0;
}