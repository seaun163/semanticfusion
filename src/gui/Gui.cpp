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

#include "Gui.h"
#include "GuiCuda.h"
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

struct ClassIdInput
{
  ClassIdInput()
    : class_id_(0) {}
  ClassIdInput(int class_id)
    : class_id_(class_id) {}
  int class_id_;
};

std::ostream& operator<< (std::ostream& os, const ClassIdInput& o){
  os << o.class_id_;
  return os;
}

std::istream& operator>> (std::istream& is, ClassIdInput& o){
  is >> o.class_id_;
  return is;
}


Gui::Gui(bool live_capture, std::vector<ClassColour> class_colour_lookup, const int segmentation_width, const int segmentation_height) 
  : width_(1280)
  , height_(980)
  , segmentation_width_(segmentation_width)
  , segmentation_height_(segmentation_height)
  , panel_(205)
  , class_colour_lookup_(class_colour_lookup)
{
  width_ += panel_;
  pangolin::Params window_params;
  window_params.Set("SAMPLE_BUFFERS", 0);
  window_params.Set("SAMPLES", 0);
  pangolin::CreateWindowAndBind("SemanticFusion", width_, height_, window_params);
  render_buffer_ = new pangolin::GlRenderBuffer(3840, 2160);
  color_texture_ = new GPUTexture(render_buffer_->width, render_buffer_->height, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, true);
  color_frame_buffer_ = new pangolin::GlFramebuffer;
  color_frame_buffer_->AttachColour(*color_texture_->texture);
  color_frame_buffer_->AttachDepth(*render_buffer_);
  s_cam_ = pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
                                      pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 1, pangolin::AxisNegY));
  pangolin::Display("cam").SetBounds(0, 1.0f, 0, 1.0f, -640 / 480.0)
    .SetHandler(new pangolin::Handler3D(s_cam_));
  // Small views along the bottom
  pangolin::Display("raw").SetAspect(640.0f/480.0f);
  pangolin::Display("pred").SetAspect(640.0f/480.0f);
  pangolin::Display("segmentation").SetAspect(640.0f/480.0f);
  pangolin::Display("multi").SetBounds(pangolin::Attach::Pix(0),1/4.0f,pangolin::Attach::Pix(180),1.0).SetLayout(pangolin::LayoutEqualHorizontal)
    .AddDisplay(pangolin::Display("pred"))
    .AddDisplay(pangolin::Display("segmentation"))
    .AddDisplay(pangolin::Display("raw"));

  // Vertical view along the side
  pangolin::Display("legend").SetAspect(640.0f/480.0f);
  pangolin::Display("vert").SetBounds(pangolin::Attach::Pix(0),1/4.0f,pangolin::Attach::Pix(180),1.0).SetLayout(pangolin::LayoutEqualVertical)
    .AddDisplay(pangolin::Display("legend"));

  // The control panel
  pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(panel_));
  pause_.reset(new pangolin::Var<bool>("ui.Pause", false, true));
  step_.reset(new pangolin::Var<bool>("ui.Step", false, false));
  reset_.reset(new pangolin::Var<bool>("ui.Reset", false, false));
  tracking_.reset(new pangolin::Var<bool>("ui.Tracking Only", false, false));
  class_view_.reset(new pangolin::Var<bool>("ui.Class Colours", false, false));
  class_choice_.reset(new pangolin::Var<ClassIdInput>("ui.Show class probs", ClassIdInput(0)));
  probability_texture_array_.reset(new pangolin::GlTextureCudaArray(224,224,GL_LUMINANCE32F_ARB));
  rendered_segmentation_texture_array_.reset(new pangolin::GlTextureCudaArray(segmentation_width_,segmentation_height_,GL_RGBA32F));

  // The gpu colour lookup
  std::vector<float> class_colour_lookup_rgb;
  for (unsigned int class_id = 0; class_id < class_colour_lookup_.size(); ++class_id) {
    class_colour_lookup_rgb.push_back(static_cast<float>(class_colour_lookup_[class_id].r)/255.0f);
    class_colour_lookup_rgb.push_back(static_cast<float>(class_colour_lookup_[class_id].g)/255.0f);
    class_colour_lookup_rgb.push_back(static_cast<float>(class_colour_lookup_[class_id].b)/255.0f);
  }
  cudaMalloc((void **)&class_colour_lookup_gpu_, class_colour_lookup_rgb.size() * sizeof(float));
  cudaMemcpy(class_colour_lookup_gpu_, class_colour_lookup_rgb.data(), class_colour_lookup_rgb.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&segmentation_rendering_gpu_,  4 * segmentation_width_ * segmentation_height_ * sizeof(float));
}

Gui::~Gui() { 
  cudaFree(class_colour_lookup_gpu_);
  cudaFree(segmentation_rendering_gpu_);
}

void Gui::preCall()
{
  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);
  glDepthFunc(GL_LESS);
  glClearColor(1.0,1.0,1.0, 0.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  width_ = pangolin::DisplayBase().v.w;
  height_ = pangolin::DisplayBase().v.h;
  pangolin::Display("cam").Activate(s_cam_);
}

void Gui::renderMap(const std::unique_ptr<ElasticFusionInterface>& map) {
  map->RenderMapToBoundGlBuffer(s_cam_,class_colours());
}

void Gui::postCall() {
  pangolin::FinishFrame();
  glFinish();
}

void Gui::displayArgMaxClassColouring(const std::string & id, float* device_ptr, int channels, const float* map_max, const int map_size,cudaTextureObject_t ids, const float threshold) {
  colouredArgMax(segmentation_width_*segmentation_height_,device_ptr,channels,class_colour_lookup_gpu_,segmentation_rendering_gpu_,map_max,map_size,ids,threshold);
  gpuErrChk(cudaGetLastError());
  gpuErrChk(cudaGetLastError());
  pangolin::CudaScopedMappedArray arr_tex(*rendered_segmentation_texture_array_.get());
  cudaMemcpyToArray(*arr_tex, 0, 0, (void*)segmentation_rendering_gpu_, sizeof(float) * 4 * segmentation_width_ * segmentation_height_, cudaMemcpyDeviceToDevice);
  gpuErrChk(cudaGetLastError());
  gpuErrChk(cudaGetLastError());
  glDisable(GL_DEPTH_TEST);
  pangolin::Display(id).Activate();
  rendered_segmentation_texture_array_->RenderToViewport(true);
  glEnable(GL_DEPTH_TEST);
}

void Gui::displayRawNetworkPredictions(const std::string & id, float* device_ptr) {
  pangolin::CudaScopedMappedArray arr_tex(*probability_texture_array_.get());
  gpuErrChk(cudaGetLastError());
  float* my_device_ptr = device_ptr + (224 * 224) * class_choice_.get()->Get().class_id_;
  cudaMemcpyToArray(*arr_tex, 0, 0, (void*)my_device_ptr, sizeof(float) * 224 * 224 , cudaMemcpyDeviceToDevice);
  gpuErrChk(cudaGetLastError());
  glDisable(GL_DEPTH_TEST);
  pangolin::Display(id).Activate();
  probability_texture_array_->RenderToViewport(true);
  glEnable(GL_DEPTH_TEST);
}

void Gui::displayImg(const std::string & id, GPUTexture * img) {
  glDisable(GL_DEPTH_TEST);
  pangolin::Display(id).Activate();
  img->texture->RenderToViewport(true);
  glEnable(GL_DEPTH_TEST);
}
