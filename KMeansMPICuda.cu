#include <cuda_runtime.h>
#include "KMeansCuda.h"

void launch_cuda() {
    // Launch GPU kernel
    // cuda_hello<<<2,4>>>();

    // cuda synch barrier
    cudaDeviceSynchronize();

}