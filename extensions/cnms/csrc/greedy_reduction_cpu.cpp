// greedy_reduction_cpu.cpp
#include <torch/extension.h>
#include <vector>

// CPU Greedy Reduction Function
void greedy_reduction_cpu_kernel(
    const int* sorted_indices,    // Shape: (N, P)
    const int* idx,              // Shape: (N, P, K)
    bool* retain,                // Shape: (N, P)
    int num_batches,
    int num_spheres,
    int num_neighbors
) {
    for (int batch = 0; batch < num_batches; ++batch) {
        // Initialize retain array for this batch
        for (int i = 0; i < num_spheres; ++i) {
            retain[batch * num_spheres + i] = true;
        }

        // Perform greedy reduction for this batch
        for (int i = 0; i < num_spheres; ++i) {
            int sphere_idx = sorted_indices[batch * num_spheres + i];
            if (!retain[batch * num_spheres + sphere_idx]) {
                continue; // Already removed
            }

            // Iterate through neighbors
            for (int j = 0; j < num_neighbors; ++j) {
                int neighbor = idx[batch * num_spheres * num_neighbors + sphere_idx * num_neighbors + j];
                if (neighbor == sphere_idx) {
                    continue; // Exclude self
                }
                retain[batch * num_spheres + neighbor] = false;
            }
        }
    }
}
