void launch_gpu_sgemm_share(
    torch::Tensor d_A ,
    torch::Tensor d_B ,
    torch::Tensor d_C ,
    int N ,
    int M ,
    int K
);