__global__ void MatAdd(float* c ,
    const float* a ,
    const float* b ,
    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = j * n + i;
    if (i < n && j < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void launch_add2(float* c ,
    const float* a ,
    const float* b ,
    int n) {
    dim3 threadPerBlock(16 , 16);
    dim3 numBlocks((n + threadPerBlock.x - 1) / threadPerBlock.x ,
        (n + threadPerBlock.y - 1) / threadPerBlock.y);
    MatAdd << <numBlocks , threadPerBlock >> > (c , a , b , n);
}
