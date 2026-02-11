#include <torch/extension.h>
#include "sgemm.h"


torch::Tensor sgemm_sum(torch::Tensor mat_A , torch::Tensor mat_B) {
    TORCH_CHECK(mat_A.is_cuda() , "mat_A must be CUDA tensor");
    TORCH_CHECK(mat_A.dtype() == torch::kFloat32 , "only float32 supported");
    TORCH_CHECK(mat_A.is_contiguous() , "mat_A must be contiguous");

    TORCH_CHECK(mat_B.is_cuda() , "mat_B must be CUDA tensor");
    TORCH_CHECK(mat_B.dtype() == torch::kFloat32 , "only float32 supported");
    TORCH_CHECK(mat_B.is_contiguous() , "mat_B must be contiguous");

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    torch::Tensor mat_C = torch::empty({ mat_A.size(0), mat_B.size(1) } , options);

    launch_gpu_sgemm_share(mat_A , mat_B , mat_C , mat_B.size(1) , mat_A.size(0) , mat_A.size(1));

    return mat_C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME , m) {
    m.def("sgemm_sum" , &sgemm_sum , "CUDA reduction sum");
}

TORCH_LIBRARY(add2 , m) {
    m.def("sgemm_sum" , sgemm_sum);
}