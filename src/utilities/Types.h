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

#ifndef TYPES_H_
#define TYPES_H_

#include <stddef.h>
#include <stdio.h>
#include <memory>
#include <string>

typedef unsigned char* ImagePtr;
typedef unsigned short* DepthPtr;

struct ClassColour {
  ClassColour() 
  : name(""), r(0), g(0), b(0) {}
  ClassColour(std::string name_, int r_, int g_, int b_) 
  : name(name_), r(r_), g(g_), b(b_) {}
  std::string name;
  int r, g, b;
};

#endif /* TYPES_H_ */
