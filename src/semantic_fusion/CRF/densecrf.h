/*
    Copyright (c) 2011, Philipp Krähenbühl
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <vector>
#include <cstdlib>

class PairwisePotential{
public:
	virtual ~PairwisePotential();
	virtual void apply( float * out_values, const float * in_values, float * tmp, int value_size ) const = 0;
};
class SemiMetricFunction{
public:
	virtual ~SemiMetricFunction();
	// For two probabilities apply the semi metric transform: v_i = sum_j mu_ij u_j
	virtual void apply( float * out_values, const float * in_values, int value_size ) const = 0;
};


class DenseCRF {
protected:
	
	// Number of variables and labels
	int N_, M_;
	float *unary_, *additional_unary_, *current_, *next_, *tmp_;
	
	// Store all pairwise potentials
	std::vector<PairwisePotential*> pairwise_;
	
	
	// Auxillary functions
	void expAndNormalize( float* out, const float* in, float scale = 1.0, float relax = 1.0 );
	
	// Don't copy this object, bad stuff will happen
	DenseCRF( DenseCRF & o ){}
public:
	// Create a dense CRF model of size N with M labels
	DenseCRF( int N, int M );
	virtual ~DenseCRF();
	// Add  a pairwise potential defined over some feature space
	// The potential will have the form:    w*exp(-0.5*|f_i - f_j|^2)
	// The kernel shape should be captured by transforming the
	// features before passing them into this function
	void addPairwiseEnergy( const float * features, int D, float w=1.0f, const SemiMetricFunction * function=NULL );
	
	// Add your own favorite pairwise potential (ownwership will be transfered to this class)
	void addPairwiseEnergy( PairwisePotential* potential );
	
	// Set the unary potential for all variables and labels (memory order is [x0l0 x0l1 x0l2 .. x1l0 x1l1 ...])
	void setUnaryEnergy( const float * unary );
	
	// Run inference and return the probabilities
	void inference( int n_iterations, float* result, float relax=1.0 );
	
	// Run MAP inference and return the map for each pixel
	void map( int n_iterations, short int* result, float relax=1.0 );
	
	// Step by step inference
	void startInference();
	void stepInference( float relax = 1.0 );

	// Run inference and return the pointer to the result
	float* runInference( int n_iterations, float relax);
};

class DenseCRF2D:public DenseCRF{
protected:
	// Width, height of the 2d grid
	int W_, H_;
public:
	// Create a 2d dense CRF model of size W x H with M labels
	DenseCRF2D( int W, int H, int M );
	virtual ~DenseCRF2D();
	// Add a Gaussian pairwise potential with standard deviation sx and sy
	void addPairwiseGaussian( float sx, float sy, float w, const SemiMetricFunction * function=NULL );
	
	// Add a Bilateral pairwise potential with spacial standard deviations sx, sy and color standard deviations sr,sg,sb
	void addPairwiseBilateral( float sx, float sy, float sr, float sg, float sb, const unsigned char * im, float w, const SemiMetricFunction * function=NULL );
};

class DenseCRF3D:public DenseCRF{
protected:
  const float spatial_stddev_;
  const float colour_stddev_;
  const float normal_stddev_;
public:
	// Create a 2d dense CRF model of size W x H with M labels
	DenseCRF3D(int N, int M, float spatial_stddev, float colour_stddev, float normal_stddev );
	virtual ~DenseCRF3D();
	// Add a Gaussian pairwise potential with standard deviation sx and sy
	void addPairwiseGaussian(const float* surfel_data, float w, const std::vector<int>& valid);
	// Add a Bilateral pairwise potential with spacial standard deviations sx, sy and color standard deviations sr,sg,sb
	void addPairwiseBilateral(const float* surfel_data, float w, const std::vector<int>& valid);
	// Add a Bilateral pairwise potential with spacial standard deviations sx, sy and color standard deviations sr,sg,sb
	void addPairwiseNormal(const float* surfel_data, float w);
};
