/*
 * This file is part of SemanticFusion.
 *
 * Copyright (C) 2017 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is SemanticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/semantic-fusion/semantic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#include <stdio.h>
#include <assert.h> 

#include <cuda_runtime.h>

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool
        abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    } 
}

__global__ 
void updateSurfelClassesKernel(const int n, float* map_surfels, const float* classes, const float* probs, const float* class_colours, const float threshold)
{
    const int surfel_size = 12;
    const int surfel_color_offset = 5;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        const int class_id = static_cast<int>(classes[id]);
        const float prob   = probs[id];
        if (class_id >= 0 && prob > threshold) {
          map_surfels[id * surfel_size + surfel_color_offset] = class_colours[class_id];
        } else {
          map_surfels[id * surfel_size + surfel_color_offset] = -1.0f;
        }
    }
}

__host__ 
void updateSurfelClasses(const int n, float* map_surfels, const float* classes, const float* probs, const float* class_colours, const float threshold)
{
    const int threads = 512;
    const int blocks = (n + threads - 1) / threads;
    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    updateSurfelClassesKernel<<<dimGrid,dimBlock>>>(n,map_surfels,classes,probs,class_colours,threshold);
    gpuErrChk(cudaGetLastError());
    gpuErrChk(cudaDeviceSynchronize());
}
