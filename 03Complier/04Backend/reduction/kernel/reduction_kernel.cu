#include <torch/extension.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

template<int BLOCK_SIZE>
__global__ void reduce_kernel(const float* in,float* out, int n){
    const int tid = threadIdx.x;
    const int bid = blockDim.x * blockIdx.x * 2 + tid;
    extern __shared__ float s_y[];

    float y = 0;
    if(bid<n) y = in[bid];
    if(bid + blockDim.x<n)y += in[bid + blockDim.x];

    s_y[tid] = y;
    
    __syncthreads();
    
    #pragma unroll
    for (int offset=blockDim.x>>1;offset>=32;offset>>=1){
        if(tid<offset){
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    
    float val = s_y[tid];
    if(tid<32){
        for(int offset=16 ;offset>0;offset>>=1){
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }

    if(tid == 0){
        atomicAdd(out,val);
    }
}

template<int BLOCK_SIZE = 256>
__global__ void reduce_kernel_optim(const float* in,float* out, int n){
    const int tid = threadIdx.x;
    const int bid = blockDim.x * blockIdx.x * 2 + tid;
    constexpr int NUM_WARPS = (BLOCK_SIZE + WARP_SIZE - 1)/ WARP_SIZE;
    
    __shared__ float s_y[NUM_WARPS];

    float sum = (bid<n)?in[bid]:0.0f;
    if(bid+blockDim.x<n) sum += in[bid+blockDim.x];

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    
    for(int mask=WARP_SIZE>>1;mask>=1;mask>>=1){
        sum += __shfl_xor_sync(0xffffffff, sum, mask);
    }
    if(lane == 0){
        s_y[warp] = sum;
    }
    __syncthreads();

    sum = (lane<NUM_WARPS?s_y[lane]:0.f);

    if(warp == 0){
        for(int mask=NUM_WARPS>>1;mask>=1;mask>>=1){
            sum += __shfl_xor_sync(0xffffffff, sum, mask);
        }
    }
    if(tid == 0){
        atomicAdd(out,sum);
    }

}

void reduce_launcher(torch::Tensor input, torch::Tensor output) {
    const int n = input.numel();
    const int block = 256;
    const int grid = (n + block * 2 - 1) / (block * 2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    reduce_kernel_optim<256><<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("[reduce_kernel] time = %.6f ms, grid=%d, block=%d\n",
           ms, grid, block);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
