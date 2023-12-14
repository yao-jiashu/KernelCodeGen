#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

/*优化方式:
 * 共享内存的使用
 * 外积法
 * 向量存取
 * 循环展开
 * 双缓冲
 * 消除bank冲突
 TODO: 换乘法指令为移位
 * */
template<
        const int BLOCK_SIZE_M,
        const int BLOCK_SIZE_N,
        const int BLOCK_SIZE_K,
        const int THREAD_SIZE_Y,
        const int THREAD_SIZE_X>
__global__ void matmul_04(
        float* __restrict__ A,
        float* __restrict__ B,
        float* __restrict__ C,
        const int M,
        const int N,
        const int K) {
    int bx = blockIdx.x; //TODO:反思，写成了blockDim.x实属不应该
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //这样写是必要的，否则不能被nvcc解释为常量，即用blockDim.y, blockDim.x不行
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    const int tid =  ty * THREAD_X_PER_BLOCK + tx;

    __shared__ float A_shared[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float B_shared[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    // TODO:忘记初始化，难顶
    double c_reg[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};

    float a_frag[2][THREAD_SIZE_Y];
    float b_frag[2][THREAD_SIZE_X];

    // 从全局内存搬运的临时寄存器
    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);
    const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_NUM_PER_BLOCK * 4);
    float ldg_a_reg[4 * ldg_num_a];
    float ldg_b_reg[4 * ldg_num_b];

    // 一些有关从全局内存分块中取数的变量,注意：一个线程一次取4个数
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    const int warpId = tid / 32;
    const int laneId = tid % 32;
    // 每个线程读4个数据，要想没有bank conflict，32/4 = 8，在组织warp tile时，需要有8个线程去读不同数据，然后4个线程去读相同的数据 32 = 8 × 4

    const int load_index_a_start = (warpId / 4) * 32 + (laneId / 4) * 4;
    const int load_index_b_start = (warpId % 4) * 16 + (laneId % 4) * 4;

    A = &A[(BLOCK_SIZE_M * by)* K];
    B = &B[BLOCK_SIZE_N * bx];
    // 从全局内存中取A分块
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
        int ldg_idx = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_idx]) = FETCH_FLOAT4(A[OFFSET(
                A_TILE_ROW_START + i,
                A_TILE_COL,
                K)]);
        A_shared[0][A_TILE_COL + 0][A_TILE_ROW_START + i] = ldg_a_reg[ldg_idx + 0];
        A_shared[0][A_TILE_COL + 1][A_TILE_ROW_START + i] = ldg_a_reg[ldg_idx + 1];
        A_shared[0][A_TILE_COL + 2][A_TILE_ROW_START + i] = ldg_a_reg[ldg_idx + 2];
        A_shared[0][A_TILE_COL + 3][A_TILE_ROW_START + i] = ldg_a_reg[ldg_idx + 3];
    }
    //从全局内存取B分块
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_K; i+= B_TILE_ROW_STRIDE) {
        FETCH_FLOAT4(B_shared[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                B_TILE_ROW_START + i,
                B_TILE_COL,
                N)]);
    }

    __syncthreads();
    //从共享内存中取a
    FETCH_FLOAT4(a_frag[0][0]) = FETCH_FLOAT4(A_shared[0][0][load_index_a_start]);
    FETCH_FLOAT4(a_frag[0][4]) = FETCH_FLOAT4(A_shared[0][0][load_index_a_start + 64]);
    // 从共享内存中取b
    FETCH_FLOAT4(b_frag[0][0]) = FETCH_FLOAT4(B_shared[0][0][load_index_b_start]);
    FETCH_FLOAT4(b_frag[0][4]) = FETCH_FLOAT4(B_shared[0][0][load_index_b_start + 64]);

    int write_buffer_id = 1;
    int tile_idx = 0;
    do {
        tile_idx += BLOCK_SIZE_K;
        //取下一个A分块到寄存器中，后面的计算不依赖于此次数据的读取，因此可以继续执行
        if (tile_idx < K) {
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i+= A_TILE_ROW_STRIDE) {
                int ldg_idx = i / A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_idx]) = FETCH_FLOAT4(A[OFFSET(
                        A_TILE_ROW_START + i,
                        A_TILE_COL + tile_idx,
                        K)]);
            }
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_K; i+= B_TILE_ROW_STRIDE) {
                int ldg_idx = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_idx]) = FETCH_FLOAT4(B[OFFSET(
                        B_TILE_ROW_START + tile_idx + i,
                        B_TILE_COL,
                        N)]);
            }
        }

        int read_buffer_id = write_buffer_id ^ 1;

        //进行小迭代， 通过i对2的余数来确定使用哪个寄存器缓冲
        // TODO:错写成 #pragma unoll, 1024×1024×1024时间变成了0.86ms，慢了三倍多,nvcc对预编译指令输入错误没有提示
        // 满了三倍多，是因为寄存器溢出，使用了local memory，导致非常大的延迟
#pragma unroll
        for (int j = 0; j < BLOCK_SIZE_K - 1; ++j) {
            // 进行下一次数据预取
            //从共享内存中取a
            FETCH_FLOAT4(a_frag[(j + 1) % 2][0]) = FETCH_FLOAT4(A_shared[read_buffer_id][j + 1][load_index_a_start]);
            FETCH_FLOAT4(a_frag[(j + 1) % 2][4]) = FETCH_FLOAT4(A_shared[read_buffer_id][j + 1][load_index_a_start + 64]);
            // 从共享内存中取b
            FETCH_FLOAT4(b_frag[(j + 1) % 2][0]) = FETCH_FLOAT4(B_shared[read_buffer_id][j + 1][load_index_b_start]);
            FETCH_FLOAT4(b_frag[(j + 1) % 2][4]) = FETCH_FLOAT4(B_shared[read_buffer_id][j + 1][load_index_b_start + 64]);


            // 执行计算
#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    c_reg[thread_y][thread_x] += a_frag[j % 2][thread_y] * b_frag[j % 2][thread_x];
                }
            }
        }

        if (tile_idx < K) {
            //将寄存器中的数据存到共享内存
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i+= A_TILE_ROW_STRIDE) {
                int ldg_idx = i / A_TILE_ROW_STRIDE * 4;
                A_shared[write_buffer_id][A_TILE_COL + 0][A_TILE_ROW_START + i] = ldg_a_reg[ldg_idx + 0];
                A_shared[write_buffer_id][A_TILE_COL + 1][A_TILE_ROW_START + i] = ldg_a_reg[ldg_idx + 1];
                A_shared[write_buffer_id][A_TILE_COL + 2][A_TILE_ROW_START + i] = ldg_a_reg[ldg_idx + 2];
                A_shared[write_buffer_id][A_TILE_COL + 3][A_TILE_ROW_START + i] = ldg_a_reg[ldg_idx + 3];
            }
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_K; i+= B_TILE_ROW_STRIDE) {
                int ldg_idx = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(B_shared[write_buffer_id][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_idx]);
            }
            //使用了双缓冲，所以只需要进行一次同步
            __syncthreads();
            write_buffer_id ^= 1;
        }

        // 指令调度，同一block同步后，所有的warp都去访问shared memory，导致计算单元闲置，然后shared memory的带宽也不够所有warp读数
        // 完成小迭代的最后一次循环计算
#pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y ++) {
#pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x ++) {
                // 最后一次，i % 2 = (BLOCK_SIZE_K - 1) % 2 = 1
                c_reg[thread_y][thread_x] += a_frag[1][thread_y] * b_frag[1][thread_x];
            }
        }

        //其实最后一次大迭代多读了，但是放到上面的if语句中就会变慢
        //从共享内存中取a
        FETCH_FLOAT4(a_frag[0][0]) = FETCH_FLOAT4(A_shared[read_buffer_id^1][0][load_index_a_start]);
        FETCH_FLOAT4(a_frag[0][4]) = FETCH_FLOAT4(A_shared[read_buffer_id^1][0][load_index_a_start + 64]);
        // 从共享内存中取b
        FETCH_FLOAT4(b_frag[0][0]) = FETCH_FLOAT4(B_shared[read_buffer_id^1][0][load_index_b_start]);
        FETCH_FLOAT4(b_frag[0][4]) = FETCH_FLOAT4(B_shared[read_buffer_id^1][0][load_index_b_start + 64]);

    } while (tile_idx < K);

    float c_reg1[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            c_reg1[i][j] = c_reg[i][j];
        }
    }

    //将结果写回到C中, 因为读取数据时不连续，写的时候要根据分块来写
    // C00, 每一行取4个数，取4行
    #pragma unroll
    for (int i = 0;i < 4; i++) {
        FETCH_FLOAT4(C[OFFSET(
                BLOCK_SIZE_M * by + load_index_a_start + i,
                BLOCK_SIZE_N * bx + load_index_b_start,
                N)]) = FETCH_FLOAT4(c_reg1[i][0]);
    }
    // C01
    #pragma unroll
    for (int i = 0;i < 4; i++) {
        FETCH_FLOAT4(C[OFFSET(
                BLOCK_SIZE_M * by + load_index_a_start + i,
                BLOCK_SIZE_N * bx + load_index_b_start + 64,
                N)]) = FETCH_FLOAT4(c_reg1[i][4]);
    }
    // C10
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        FETCH_FLOAT4(C[OFFSET(
                BLOCK_SIZE_M * by + load_index_a_start + i + 64,
                BLOCK_SIZE_N * bx + load_index_b_start,
                N)]) = FETCH_FLOAT4(c_reg1[4 + i][0]);
    }
    // C11
    #pragma unroll
    for (int i = 0;i < 4; i++) {
        FETCH_FLOAT4(C[OFFSET(
                BLOCK_SIZE_M * by + load_index_a_start + i + 64,
                BLOCK_SIZE_N * bx + load_index_b_start + 64,
                N)]) = FETCH_FLOAT4(c_reg1[4 + i][4]);
    }
}