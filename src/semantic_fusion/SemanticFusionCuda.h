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

#include <cuda_runtime.h>

void fuseSemanticProbabilities(cudaTextureObject_t ids, const int ids_width, const int ids_height, 
                          const float* probabilities, const int prob_width, const int prob_height, 
                          const int prob_channels,float* map_table, float* map_max,
                          const int map_size);

void updateProbabilityTable(int* deleted_ids, const int num_deleted, const int current_table_size,
                            float const* probability_table, const int prob_width, const int prob_height, 
                          const int new_prob_width, float* new_probability_table, 
                          float const* map_table, float* new_map_table);

void renderProbabilityMap(cudaTextureObject_t ids, const int ids_width, const int ids_height, 
                          const float* probability_table, const int prob_width, const int prob_height, 
                          float* rendered_probabilities);


void updateMaxClass(const int n, const float* probabilities, const int classes,
                    float* map_max, const int map_size);
