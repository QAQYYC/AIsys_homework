import sys
sys.path.append("build/lib.linux-x86_64-cpython-310")
import torch
from torch.utils.cpp_extension import load
print(torch.cuda.is_available())


add2 = load(
    name="add2",
    sources=[
        "03Complier/04Backend/add2/kernel/add2_kernel.cu",
        "03Complier/04Backend/add2/kernel/add2_ops.cpp",
    ],
    extra_include_paths=["03Complier/04Backend/add2/include"],
    verbose=True,
)
import time

if __name__ == "__main__":
    a = time.time()

    input_a = torch.ones((48,48),dtype=torch.float32,device='cuda:0')
    input_b = torch.ones((48,48),dtype=torch.float32,device='cuda:0')
    input_c = torch.zeros((48,48),dtype=torch.float32,device='cuda:0')

    add2.torch_launch_add2(input_c, input_a, input_b, 48)
    print(input_c)
