#include <torch/extension.h>

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
        out[blockIdx.x] = val;
    }
}

void reduce_launcher(torch::Tensor input , torch::Tensor output) {
    const int n = input.numel();

    const int block = 256;
    const int grid = (n + block * 2 - 1) / (block * 2);

    reduce_kernel<256> << <grid , block , block * sizeof(float)>> > (
        input.data_ptr<float>() ,
        output.data_ptr<float>() ,
        n
        );
}

