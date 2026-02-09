#include <torch/extension.h>
#include "reduction.h"
#include <vector>


torch::Tensor reduce_sum(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() , "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32 , "only float32 supported");
    TORCH_CHECK(input.is_contiguous() , "input must be contiguous");

    auto cur = input;
    auto tmp = torch::empty(
        { (cur.numel() + 511) / 512 } ,
        cur.options()
    );

    while (cur.numel() > 1) {
        reduce_launcher(cur , tmp);
        cur = tmp;
        tmp = torch::empty(
            { (cur.numel() + 511) / 512 } ,
            cur.options()
        );
    }

    return cur;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME , m) {
    m.def("reduce_sum" , &reduce_sum , "CUDA reduction sum");
}

TORCH_LIBRARY(add2 , m) {
    m.def("reduce_sum" , reduce_sum);
}