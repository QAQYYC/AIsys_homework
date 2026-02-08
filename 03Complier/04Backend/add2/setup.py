from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='add2',
    ext_modules=[
        CUDAExtension(
            name='add2',
            sources=[
                '03Complier/04Backend/kernel/add2_kernel.cu',
                '03Complier/04Backend/kernel/add2_ops.cpp',
            ],
            include_dirs=[
                '03Complier/04Backend/include'
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)