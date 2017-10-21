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

#ifndef GUI_H_
#define GUI_H_
#include <iostream>
#include <memory>

#include <cuda.h>

#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/gldraw.h>

#include <map_interface/ElasticFusionInterface.h>
#include <utilities/Types.h>

struct ClassIdInput;

class Gui {
public:
  //enum SelectProbabilityMap {Books,Chairs,Floor};
  Gui(bool live_capture,std::vector<ClassColour> class_colour_lookup, const int segmentation_width, const int segmentation_height);
  virtual ~Gui();

  void preCall();
  void renderMap(const std::unique_ptr<ElasticFusionInterface>& map);
  void postCall();
  void displayArgMaxClassColouring(const std::string & id, float* device_ptr, int channels,const float* map, const int map_size, cudaTextureObject_t ids, const float threshold);
  void displayRawNetworkPredictions(const std::string & id, float* device_ptr);
  void displayImg(const std::string & id, GPUTexture * img);

  bool reset() const { return pangolin::Pushed(*reset_.get()); }
  bool paused() const { return *pause_.get(); }
  bool step() const { return pangolin::Pushed(*step_.get()); }
  bool tracking() const { return *tracking_.get(); }
  bool class_colours() const { return *class_view_.get(); }

private:
  int width_;
  int height_;
  const int segmentation_width_;
  const int segmentation_height_;
  int panel_;
  std::vector<ClassColour> class_colour_lookup_;
  float* class_colour_lookup_gpu_;
  float* segmentation_rendering_gpu_;

  std::unique_ptr<pangolin::Var<bool>> reset_;
  std::unique_ptr<pangolin::Var<bool>> pause_;
  std::unique_ptr<pangolin::Var<bool>> step_;
  std::unique_ptr<pangolin::Var<bool>> tracking_;
  std::unique_ptr<pangolin::Var<bool>> class_view_;
  std::unique_ptr<pangolin::Var<ClassIdInput>> class_choice_;
  std::unique_ptr<pangolin::GlTextureCudaArray> probability_texture_array_;
  std::unique_ptr<pangolin::GlTextureCudaArray> rendered_segmentation_texture_array_;
  pangolin::GlRenderBuffer* render_buffer_;
  pangolin::GlFramebuffer* color_frame_buffer_;
  GPUTexture* color_texture_;
  pangolin::OpenGlRenderState s_cam_;
};


#endif /* GUI_H_ */
