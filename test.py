import sys
sys.path.append("build/lib.linux-x86_64-cpython-310")
import torch
from torch.utils.cpp_extension import load
# print(torch.cuda.is_available())


# add2 = load(
#     name="add2",
#     sources=[
#         "03Complier/04Backend/add2/kernel/add2_kernel.cu",
#         "03Complier/04Backend/add2/kernel/add2_ops.cpp",
#     ],
#     extra_include_paths=["03Complier/04Backend/add2/include"],
#     verbose=True,
# )

# reduce = load(
#     name='reduce',
#     sources=[
#         "03Complier/04Backend/reduction/kernel/reduction_kernel.cu",
#         "03Complier/04Backend/reduction/kernel/reduction_ops.cpp"
#     ],
#     extra_include_paths=["03Complier/04Backend/reduction/include"],
#     verbose=True
# )

sgemm = load(
    name="sgemm",
    sources=[
        "03Complier/04Backend/SGEMM/kernel/sgemm_kernel.cu",
        "03Complier/04Backend/SGEMM/kernel/sgemm_ops.cpp"
    ],
    extra_include_paths=["03Complier/04Backend/SGEMM/include"],
    verbose=True,
    extra_cuda_cflags=["-G"]
)
import time

if __name__ == "__main__":
    a = time.time()

    # input_a = torch.ones((100000),dtype=torch.float32,device='cuda:0')
    # output_a = reduce.reduce_sum(input_a)

    mat_A = torch.ones((100,128),dtype = torch.float32, device='cuda:0')
    mat_B = torch.ones((128,100),dtype = torch.float32, device='cuda:0')

    output_a = sgemm.sgemm_sum(mat_A, mat_B)

    print(output_a)
    print(output_a.shape)
