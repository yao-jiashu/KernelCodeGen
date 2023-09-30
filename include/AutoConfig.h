#pragma once
#include "IR/MLIRExtension.h"

namespace KernelCodegen {

struct GEMMConfig {
    GEMMConfig(std::initializer_list<int>&& problem_size_, 
                std::initializer_list<int>&& vector_len_ = {4, 4},
                std::initializer_list<int>&& block_workload_ = {128, 128, 8},
                std::initializer_list<int>&& thread_workload_ = {8, 8},
                std::initializer_list<int>&& thread_per_block_ = {16, 16}) : 
                problem_size(problem_size_), 
                vector_len(vector_len_),
                block_workload(block_workload_),
                thread_workload(thread_workload_),
                thread_per_block(thread_per_block_) {}
    std::vector<int> problem_size {4096, 4096, 4096};
    std::vector<int> block_workload {128, 128, 8};
    std::vector<int> thread_workload {8, 8};
    std::vector<int> thread_per_block {16, 16};

    std::vector<int> vector_len {4, 4};

    int lda_regs_per_thread {-1};
    int ldb_regs_per_thread {-1};

    int get_lda_regs_per_thread() {

        int threads_num_block = thread_per_block[0] * thread_per_block[1];

        if (lda_regs_per_thread == -1) {
            lda_regs_per_thread = block_workload[0] * block_workload[2] /
                                    (threads_num_block * vector_len[0]);
        }
        return lda_regs_per_thread;
    }

    int get_ldb_regs_per_thread() {
        int threads_num_block = thread_per_block[0] * thread_per_block[1];

        if (ldb_regs_per_thread == -1) {
            ldb_regs_per_thread = block_workload[2] * block_workload[2] /
                                    (threads_num_block * vector_len[1]);
        }
        return ldb_regs_per_thread;
    }

};


}