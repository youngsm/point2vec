# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import find_packages

setup(
    name="cnms",
    version="0.0",
    description="Centrality-Based Non-Maximum Suppression (C-NMS)",
    author="Sam Young",
    author_email="youngsam@stanford.edu",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "dist",
            "cnms.egg-info",
        )
    ),
    ext_modules=[
        CUDAExtension(
            name="cnms._ext",
            sources=[
                "csrc/greedy_reduction.cpp",
                "csrc/greedy_reduction_cuda.cu",
                "csrc/greedy_reduction_cpu.cpp",
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch",
        "pytorch3d",
        "ninja"
    ],
)
