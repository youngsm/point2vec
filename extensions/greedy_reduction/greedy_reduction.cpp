// greedy_reduction.cpp
#include <torch/extension.h>
#include <vector>

// Forward declarations of CPU and CUDA functions
void greedy_reduction_cpu_kernel(
    const int* sorted_indices,
    const int* idx,
    bool* retain,
    int num_batches,
    int num_spheres,
    int num_neighbors
);

void launch_greedy_reduction_cuda_kernel(
    const int* sorted_indices,
    const int* idx,
    bool* retain,
    int num_batches,
    int num_spheres,
    int num_neighbors
);

// C++ Interface Function
torch::Tensor greedy_reduction(
    torch::Tensor sorted_indices,
    torch::Tensor idx
) {
    // Check that input tensors are 2D and 3D respectively
    TORCH_CHECK(sorted_indices.dim() == 2, "sorted_indices must be a 2D tensor");
    TORCH_CHECK(idx.dim() == 3, "idx must be a 3D tensor");

    // Ensure that the batch sizes and sphere counts match
    TORCH_CHECK(sorted_indices.size(0) == idx.size(0),
                "Batch size of sorted_indices and idx must match");
    TORCH_CHECK(sorted_indices.size(1) == idx.size(1),
                "Number of spheres in sorted_indices and idx must match");

    // Ensure inputs are on the same device
    TORCH_CHECK(sorted_indices.device() == idx.device(),
                "sorted_indices and idx must be on the same device");

    // Determine device type
    bool is_cuda = sorted_indices.is_cuda();

    // Get dimensions
    int num_batches = sorted_indices.size(0);
    int num_spheres = sorted_indices.size(1);
    int num_neighbors = idx.size(2);

    // Initialize retain tensor
    auto retain = torch::ones({num_batches, num_spheres}, torch::dtype(torch::kBool).device(sorted_indices.device()));

    if (is_cuda) {
        // Ensure tensors are contiguous and of type int32
        TORCH_CHECK(sorted_indices.dtype() == torch::kInt32 || sorted_indices.dtype() == torch::kInt64,
                    "sorted_indices must be of type int32 or int64");
        TORCH_CHECK(idx.dtype() == torch::kInt32 || idx.dtype() == torch::kInt64,
                    "idx must be of type int32 or int64");

        // Convert tensors to int32 if they are int64 for better performance
        if (sorted_indices.dtype() == torch::kInt64) {
            sorted_indices = sorted_indices.to(torch::kInt32);
        }
        if (idx.dtype() == torch::kInt64) {
            idx = idx.to(torch::kInt32);
        }

        // Get raw pointers
        const int* sorted_indices_ptr = sorted_indices.data_ptr<int>();
        const int* idx_ptr = idx.data_ptr<int>();
        bool* retain_ptr = retain.data_ptr<bool>();

        // Launch CUDA kernel
        launch_greedy_reduction_cuda_kernel(
            sorted_indices_ptr,
            idx_ptr,
            retain_ptr,
            num_batches,
            num_spheres,
            num_neighbors
        );
    }
    else {
        // Ensure tensors are contiguous and of type int32
        TORCH_CHECK(sorted_indices.dtype() == torch::kInt32 || sorted_indices.dtype() == torch::kInt64,
                    "sorted_indices must be of type int32 or int64");
        TORCH_CHECK(idx.dtype() == torch::kInt32 || idx.dtype() == torch::kInt64,
                    "idx must be of type int32 or int64");

        // Convert tensors to int32 if they are int64 for better performance
        if (sorted_indices.dtype() == torch::kInt64) {
            sorted_indices = sorted_indices.to(torch::kInt32);
        }
        if (idx.dtype() == torch::kInt64) {
            idx = idx.to(torch::kInt32);
        }

        // Get raw pointers
        const int* sorted_indices_ptr = sorted_indices.data_ptr<int>();
        const int* idx_ptr = idx.data_ptr<int>();
        bool* retain_ptr = retain.data_ptr<bool>();

        // Launch CPU kernel
        greedy_reduction_cpu_kernel(
            sorted_indices_ptr,
            idx_ptr,
            retain_ptr,
            num_batches,
            num_spheres,
            num_neighbors
        );
    }

    return retain;
}

// Binding the C++ Interface Function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("greedy_reduction", &greedy_reduction, "Greedy Reduction (CPU and CUDA)");
}
