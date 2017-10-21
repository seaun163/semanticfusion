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
void colouredArgMaxKernel(int n, float const* probabilities,  const int num_classes, float const* color_lookup, float* colour, float const* map_max, const int map_size,cudaTextureObject_t ids, const float threshold)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        const int y = id / 640;
        const int x = id - (y * 640);
        const int start_windowx = (x - 1) > 0 ? (x - 1) : 0;
        const int start_windowy = (y - 1) > 0 ? (y - 1) : 0;
        const int end_windowx = (x + 1) < 640 ? (x + 1) : 639;
        const int end_windowy = (y + 1) < 480 ? (y + 1) : 479;

        int max_class_id = -1;
        float max_class_prob = threshold;
        for (int sx = start_windowx; sx <= end_windowx; ++sx) {
            for (int sy = start_windowy; sy <= end_windowy; ++sy) {
                const int surfel_id = tex2D<int>(ids,sx,sy);
                if (surfel_id > 0) {
                    float const* id_probabilities = map_max + surfel_id;
                    if (id_probabilities[map_size] > max_class_prob) {
                        max_class_id = static_cast<int>(id_probabilities[0]);
                        max_class_prob = id_probabilities[map_size];
                    }
                }
            }
        }

        float* local_colour = colour + (id * 4);
        if (max_class_id >= 0) {
            local_colour[0] = color_lookup[max_class_id * 3 + 0];
            local_colour[1] = color_lookup[max_class_id * 3 + 1];
            local_colour[2] = color_lookup[max_class_id * 3 + 2];
            local_colour[3] = 1.0f;
        } else {
            local_colour[0] = 0.0;
            local_colour[1] = 0.0;
            local_colour[2] = 0.0;
            local_colour[3] = 1.0f;
        }
    }
}

__host__
void colouredArgMax(int n, float const * probabilities,  const int num_classes, float const* color_lookup, float* colour, float const * map, const int map_size,cudaTextureObject_t ids, const float threshold)
{
    const int threads = 512;
    const int blocks = (n + threads - 1) / threads;
    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    colouredArgMaxKernel<<<dimGrid,dimBlock>>>(n,probabilities,num_classes,color_lookup,colour,map,map_size,ids,threshold);
    gpuErrChk(cudaGetLastError());
    gpuErrChk(cudaDeviceSynchronize());
}
