# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="greedy_reduction",
    version="0.1",
    description="Greedy Reduction for Sphere Packing",
    author="Sam Young",
    author_email="youngsam@stanford.edu",
    ext_modules=[
        CUDAExtension(
            name="greedy_reduction_extension",
            sources=[
                "greedy_reduction.cpp",
                "greedy_reduction_cuda.cu",
                "greedy_reduction_cpu.cpp",
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
