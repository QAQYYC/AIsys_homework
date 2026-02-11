#include<cstdio>
#include<torch/extension.h>


// void cpu_sgemm(float* a, float* b, float* c, int N, int M,int K){
//     for(int i=0;i<M;i++){
//         for(int j=0;j<N;j++){
//             float temp=0;
//             for(int p=0;p<K;p++){
//                 temp += a[i*K+p] * b[p*N+j];
//             }
//             c[i*N+j]=temp;
//         }
//     }
// }

template<int BLOCK_SIZE, int TILE_K>
__global__ void gpu_sgemm_share(
    float* a, float* b, float* c,
    int N, int M, int K)
{
    __shared__ float a_shared[BLOCK_SIZE][TILE_K];
    __shared__ float b_shared[TILE_K][BLOCK_SIZE];

    int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int col = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    float temp = 0.0f;

    for(int tile = 0; tile < K; tile += blockDim.y){
        if(row < M && tile + threadIdx.y < K)
            a_shared[threadIdx.x][threadIdx.y + tile] =
                a[row * K + tile + threadIdx.y];

        if(col < N && tile + threadIdx.x < K)
            b_shared[threadIdx.x + tile][threadIdx.y] =
                b[(tile + threadIdx.x) * N + col];
    }

    __syncthreads();
            
    for(int k = 0; k < TILE_K; k++){
        temp += a_shared[threadIdx.x][k] *
                b_shared[k][threadIdx.y];
        
    }
    if(row < M && col < N)
        c[row * N + col] = temp;
}

void launch_gpu_sgemm_share(
    torch::Tensor d_A,
    torch::Tensor d_B,
    torch::Tensor d_C,
    int N,
    int M,
    int K
) {
    constexpr int BLOCK_SIZE = 32;
    constexpr int TILE_K = 128;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    gpu_sgemm_share<BLOCK_SIZE, TILE_K><<<grid, block>>>(
        d_A.data_ptr<float>(), d_B.data_ptr<float>(), d_C.data_ptr<float>(),
        N, M, K
    );

    // 强烈建议：kernel launch 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}