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

#include "densecrf.h"
#include "fastmath.h"
#include "permutohedral.h"
#include "util.h"
#include <cmath>
#include <cstring>

PairwisePotential::~PairwisePotential() {
}

SemiMetricFunction::~SemiMetricFunction() {
}

class PottsPotential: public PairwisePotential {
protected:
	Permutohedral lattice_;
	PottsPotential( const PottsPotential&o ){}
	int N_;
	float w_;
	float *norm_;
public:
	~PottsPotential(){
		deallocate( norm_ );
	}
	PottsPotential(const float* features, int D, int N, float w, bool per_pixel_normalization=true) :N_(N), w_(w) {
		lattice_.init( features, D, N );
		norm_ = allocate( N );
		for ( int i=0; i<N; i++ )
			norm_[i] = 1;
		// Compute the normalization factor
		lattice_.compute( norm_, norm_, 1 );
		// use a per pixel normalization
		for ( int i=0; i<N; i++ )
	    norm_[i] = 1.f / (norm_[i]+1e-20f);
	}
	void apply(float* out_values, const float* in_values, float* tmp, int value_size) const {
		lattice_.compute( tmp, in_values, value_size );
		for ( int i=0,k=0; i<N_; i++ )
			for ( int j=0; j<value_size; j++, k++ ) {
				out_values[k] += w_*norm_[i]*tmp[k];
			}
		/*
		// Added by John - this is the form in the paper, it results in an almost
		// identical output as the code they gave, but it's much slower
		// The slight change is due entirely to removing the -in_values, i.e. the
		// component of the message which is coming from the vertex itself - this is
		// included in the gaussian convolution
		for ( int i=0,k=0; i<N_; i++ ) {
			for ( int j=0; j<value_size; j++, k++ ) {
			  int index = i * value_size;
			  for ( int l=0; l<value_size; l++ ) {
			    if (j != l) {
				    out_values[k] -= w_*norm_[i]*(tmp[index + l] - in_values[index + l]);
				  }
				}
			}
		}
		*/
	}
};

/////////////////////////////
/////  Alloc / Dealloc  /////
/////////////////////////////
DenseCRF::DenseCRF(int N, int M) : N_(N), M_(M) {
	unary_ = allocate( N_*M_ );
	current_ = allocate( N_*M_ );
	next_ = allocate( N_*M_ );
	tmp_ = allocate( 2*N_*M_ );
}

DenseCRF::~DenseCRF() {
	deallocate( unary_ );
	deallocate( current_ );
	deallocate( next_ );
	deallocate( tmp_ );
	for( unsigned int i=0; i<pairwise_.size(); i++ )
		delete pairwise_[i];
}

/////////////////////////////////
/////  Pairwise Potentials  /////
/////////////////////////////////
void DenseCRF::addPairwiseEnergy (const float* features, int D, float w, const SemiMetricFunction * function) {
	addPairwiseEnergy( new PottsPotential( features, D, N_, w ) );
}

void DenseCRF::addPairwiseEnergy ( PairwisePotential* potential ){
	pairwise_.push_back( potential );
}

DenseCRF2D::DenseCRF2D(int W, int H, int M) : DenseCRF(W*H,M), W_(W), H_(H) {
}

DenseCRF2D::~DenseCRF2D() {
}

void DenseCRF2D::addPairwiseGaussian ( float sx, float sy, float w, const SemiMetricFunction * function ) {
	float * feature = new float [N_*2];
	for( int j=0; j<H_; j++ )
		for( int i=0; i<W_; i++ ){
			feature[(j*W_+i)*2+0] = i / sx;
			feature[(j*W_+i)*2+1] = j / sy;
		}
	addPairwiseEnergy( feature, 2, w, function );
	delete [] feature;
}

void DenseCRF2D::addPairwiseBilateral ( float sx, float sy, float sr, float sg, float sb, const unsigned char* im, float w, const SemiMetricFunction * function ) {
	float * feature = new float [N_*5];
	for( int j=0; j<H_; j++ )
		for( int i=0; i<W_; i++ ){
			feature[(j*W_+i)*5+0] = i / sx;
			feature[(j*W_+i)*5+1] = j / sy;
			feature[(j*W_+i)*5+2] = im[(i+j*W_)*3+0] / sr;
			feature[(j*W_+i)*5+3] = im[(i+j*W_)*3+1] / sg;
			feature[(j*W_+i)*5+4] = im[(i+j*W_)*3+2] / sb;
		}
	addPairwiseEnergy( feature, 5, w, function );
	delete [] feature;
}

DenseCRF3D::DenseCRF3D(int N, int M, float spatial_stddev, float colour_stddev, float normal_stddev) 
  : DenseCRF(N,M)
  , spatial_stddev_(spatial_stddev)
  , colour_stddev_(colour_stddev)
  , normal_stddev_(normal_stddev) {
}

DenseCRF3D::~DenseCRF3D() {
}

void DenseCRF3D::addPairwiseGaussian (const float* surfel_data, float w, const std::vector<int>& valid) {
	const int features = 6;
	const int surfel_size = 12;
	float * feature = new float [N_*features];
	for (int i=0; i<N_; i++) {
	  const int id = valid[i];
	  int idx = 0;
		feature[(i*features)+(idx++)] = surfel_data[(id * surfel_size) + 0] / spatial_stddev_;
		feature[(i*features)+(idx++)] = surfel_data[(id * surfel_size) + 1] / spatial_stddev_;
	  feature[(i*features)+(idx++)] = surfel_data[(id * surfel_size) + 2] / spatial_stddev_;
		feature[(i*features)+(idx++)] = surfel_data[(id * surfel_size) + 8] / normal_stddev_;
		feature[(i*features)+(idx++)] = surfel_data[(id * surfel_size) + 9] / normal_stddev_;
		feature[(i*features)+(idx++)] = surfel_data[(id * surfel_size) + 10] / normal_stddev_;
		/*
    float encoded_colour = surfel_data[(i * surfel_size) + 4];
    unsigned char* colour = reinterpret_cast<unsigned char*>(&encoded_colour);
		feature[(i*features)+6] = static_cast<int>(colour[0]) / colour_stddev_;
		feature[(i*features)+7] = static_cast<int>(colour[1])/ colour_stddev_;
		feature[(i*features)+8] = static_cast<int>(colour[2])/ colour_stddev_;
		*/
	}
	addPairwiseEnergy(feature, features, w, NULL);
	delete [] feature;
}

void DenseCRF3D::addPairwiseBilateral ( const float* surfel_data, float w, const std::vector<int>& valid) {
	const int features = 6;
	float * feature = new float [N_*features];
	const int surfel_size = 12;
	for (int i=0; i<N_; i++) {
	  //Spatial filtering
	  const int id = valid[i];
	  int idx = 0;
		feature[(i*features)+(idx++)] = surfel_data[(id * surfel_size) + 0] / spatial_stddev_;
		feature[(i*features)+(idx++)] = surfel_data[(id * surfel_size) + 1] / spatial_stddev_;
		feature[(i*features)+(idx++)] = surfel_data[(id * surfel_size) + 2] / spatial_stddev_;
		/*
		feature[(i*features)+(idx++)] = surfel_data[(i * surfel_size) + 8] / normal_stddev_;
		feature[(i*features)+(idx++)] = surfel_data[(i * surfel_size) + 9] / normal_stddev_;
		feature[(i*features)+(idx++)] = surfel_data[(i * surfel_size) + 10] / normal_stddev_;
		*/
		//Colour filtering
    float encoded_colour = surfel_data[id * surfel_size + 4];
    int colour = static_cast<int>(encoded_colour);
    int r = static_cast<int>(colour >> 16 & 0xFF);
    int g = static_cast<int>(colour >> 8 & 0xFF);
    int b = static_cast<int>(colour & 0xFF);
		feature[(i*features)+(idx++)] = static_cast<float>(r) / colour_stddev_;
		feature[(i*features)+(idx++)] = static_cast<float>(g) / colour_stddev_;
		feature[(i*features)+(idx++)] = static_cast<float>(b) / colour_stddev_;
  }
	addPairwiseEnergy(feature, features, w, NULL);
	delete [] feature;
}

void DenseCRF3D::addPairwiseNormal ( const float* surfel_data, float w) {
  /*
	float * feature = new float [N_*3];
	const int surfel_size = 12;
	for (int i=0; i<N_; i++) {
	  // Normal filtering
		feature[i] = surfel_data[(i * surfel_size) + 8] / normal_stddev_;
		feature[i] = surfel_data[(i * surfel_size) + 9] / normal_stddev_;
		feature[i] = surfel_data[(i * surfel_size) + 10] / normal_stddev_;
  }
	addPairwiseEnergy(feature, 3, w, NULL);
	delete [] feature;
	*/
}

//////////////////////////////
/////  Unary Potentials  /////
//////////////////////////////
void DenseCRF::setUnaryEnergy ( const float* unary ) {
	memcpy( unary_, unary, N_*M_*sizeof(float) );
}

///////////////////////
/////  Inference  /////
///////////////////////
void DenseCRF::inference ( int n_iterations, float* result, float relax ) {
	// Run inference
	float * prob = runInference( n_iterations, relax );
	// Copy the result over
	for( int i=0; i<N_; i++ )
		memcpy( result+i*M_, prob+i*M_, M_*sizeof(float) );
}

void DenseCRF::map ( int n_iterations, short* result, float relax ) {
	// Run inference
	float * prob = runInference( n_iterations, relax );
	
	// Find the map
	for( int i=0; i<N_; i++ ){
		const float * p = prob + i*M_;
		// Find the max and subtract it so that the exp doesn't explode
		float mx = p[0];
		int imx = 0;
		for( int j=1; j<M_; j++ )
			if( mx < p[j] ){
				mx = p[j];
				imx = j;
			}
		result[i] = imx;
	}
}

float* DenseCRF::runInference( int n_iterations, float relax ) {
	startInference();
	for (int it=0; it<n_iterations; it++)
		stepInference(relax);
	return current_;
}

void DenseCRF::expAndNormalize ( float* out, const float* in, float scale, float relax ) {
	float *V = new float[ N_+10 ];
	for (int i=0; i<N_; i++) {
		const float * b = in + i*M_;
		// Find the max and subtract it so that the exp doesn't explode
		float mx = scale*b[0];
		for( int j=1; j<M_; j++ )
			if( mx < scale*b[j] )
				mx = scale*b[j];
		float tt = 0;
		for( int j=0; j<M_; j++ ){
			V[j] = fast_exp( scale*b[j]-mx );
			tt += V[j];
		}
		// Make it a probability
		for( int j=0; j<M_; j++ )
			V[j] /= tt;
		
		float * a = out + i*M_;
		for( int j=0; j<M_; j++ )
			if (relax == 1)
				a[j] = V[j];
			else
				a[j] = (1-relax)*a[j] + relax*V[j];
	}
	delete[] V;
}

void DenseCRF::startInference(){
	// Initialize using the unary energies
	expAndNormalize( current_, unary_, -1 );
}

void DenseCRF::stepInference( float relax ){
	for( int i=0; i<N_*M_; i++ )
		next_[i] = -unary_[i];
	// Add up all pairwise potentials
	for( unsigned int i=0; i<pairwise_.size(); i++ )
		pairwise_[i]->apply( next_, current_, tmp_, M_ );
	// Exponentiate and normalize
	expAndNormalize( current_, next_, 1.0, relax );
}
