/*
 * COPYRIGHT
 * 
 * All contributions by the University of California:
 * Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
 * All rights reserved.
 * 
 * All other contributions:
 * Copyright (c) 2014, 2015, the respective contributors
 * All rights reserved.
 * 
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 * 
 * LICENSE
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met: 
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer. 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution. 
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef ARRAY_H_
#define ARRAY_H_

#include <stdio.h>
#include <memory>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>

#include <glog/logging.h>
#include <caffe/caffe.hpp>

inline void MallocHost(void** ptr, size_t size, bool* use_cuda) {
  CUDA_CHECK(cudaMallocHost(ptr, size));
  *use_cuda = true;
  return;
}

inline void FreeHost(void* ptr, bool use_cuda) {
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
  free(ptr);
}


class SyncedMemory {
 public:
  SyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void set_gpu_data(void* data);
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; }
  size_t size() { return size_; }

  void async_gpu_push(const cudaStream_t& stream);

 private:
  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int gpu_device_;
}; 

class Array3D {
 public:
  explicit Array3D(const int channels, const int height, const int width) 
    : data_()
    , capacity_(0) 
  {
    Reshape(channels,height,width);
  }

  void Reshape(const int channels, const int height, const int width);

  inline int count() const { return height_ * width_ * channels_; }
  inline int channels() const { return channels_; }
  inline int height() const { return height_; }
  inline int width() const { return width_; }
  inline int offset(const int c, const int h, const int w) const {
    return ((c * height() + h) * width() + w);
  }

  const float* cpu_data() const {  
    return (const float*) data_->cpu_data(); 
  }
  float* mutable_cpu_data() { 
    return static_cast<float*>(data_->mutable_cpu_data()); 
  }
  const float* gpu_data() const {  
    return (const float*)data_->gpu_data(); 
  }
  float* mutable_gpu_data() { 
    return static_cast<float*>(data_->mutable_gpu_data()); 
  }

 protected:
  std::shared_ptr<SyncedMemory> data_;
  int capacity_;
  int channels_;
  int height_;
  int width_;
}; 

#endif /* ARRAY_H_ */
