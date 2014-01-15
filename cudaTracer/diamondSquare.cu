#include "diamondSquare.cuh"

#include "device_launch_parameters.h"
#include <stdio.h>

#include <stdlib.h>
#include <string.h>

#include <curand_kernel.h>
#include <curand_normal.h>

#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <sstream>

#include "globals.h"

// http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { throw_on_cuda_error((ans), __FILE__, __LINE__); }
//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true){
	if (code != cudaSuccess){ 
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

inline void throw_on_cuda_error(cudaError_t code, const char *file, int line)
{
  if(code != cudaSuccess)
  {
	std::stringstream ss;
	ss << file << "(" << line << ")";
	std::string file_and_line;
	ss >> file_and_line;
	throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
  }
}

//texture<float, 2, cudaReadModeElementType> texRef;

enum colors
{
	RED, GREEN, BLUE, ALPHA
};

// RNG init kernel
__global__ void initRNG(curandState *const rngStates,
						const unsigned int seed)
{
	// Determine thread ID
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	tid = x*gridDim.x + y;

	// Initialise the RNG
	curand_init( seed, tid, 0, &rngStates[tid] );
}

__device__ float rand( curandState* const localState ){
	return curand_uniform( localState );
	//return 0.5f;
}

__device__ float* getPixel( unsigned char *surface, size_t pitch,
	int x, int y, int width, int height )
{
	x = x<0				? x+(width-1)  : x;
	x = x>(width-1)		? x-(width-1)  : x;
	y = y<0				? y+(height-1) : y;
	y = y>(height-1)	? y-(height-1) : y;

	// get a pointer to the pixel at (x,y)
	return (float *)(surface + y*pitch) + 4*x;
}


__device__ float getColOrRand( unsigned char *surface, size_t pitch,
	curandState* const localState, int2 pixel, int width, int height, int idx ){

	/*if( pixel.x > width-1 ){
		pixel.x = pixel.x-(width-1);
	} else if( pixel.x < 0 ){
		pixel.x = width*rand(localState);
	}
	if( pixel.y > height-1 ){
		pixel.y = pixel.y-(height-1);
	} else if( pixel.y < 0 ){
		pixel.y = height*rand(localState);
	}*/

	float* pixPtr = getPixel( surface, pitch, pixel.x, pixel.y, width, height);
	float pixCol = pixPtr[idx];
	//return pixCol > EPSILON ? pixCol : rand( localState )*INIT_RAND_FAC;

	//return pixCol < EPSILON ? pixCol : blockIdx.x/32.0f;
	//return pixCol > EPSILON ? pixCol : blockIdx.x/32.0f;
	return pixCol > EPSILON ? pixCol : -0.1f; // blood mode
	
	//return pixCol;
}

__device__ float createColor( unsigned char *surface, size_t pitch,
	curandState* const localState, int2 a, int2 b, int2 c, int2 d, int width, int height, int idx ){
	float aCol = getColOrRand( surface, pitch, localState, a, width, height, idx );
	float bCol = getColOrRand( surface, pitch, localState, b, width, height, idx );
	float cCol = getColOrRand( surface, pitch, localState, c, width, height, idx );
	float dCol = getColOrRand( surface, pitch, localState, d, width, height, idx );
	float finalCol = (aCol + bCol + cCol + dCol) * 0.25f;
	return finalCol;// + ((rand(localState)*2 - 1)) * step * RAND_FAC;
}

__device__ void diamond( unsigned char *surface, size_t pitch, float* pixel,
	curandState* const rngStates, int x, int y, int step, int width, int height ){

	/*float* rb = getPixel( surface, pitch, x+step, y+step, width, height);
	float* lt = getPixel( surface, pitch, x-step, y-step, width, height);
	float* lb = getPixel( surface, pitch, x-step, y+step, width, height);
	float* rt = getPixel( surface, pitch, x+step, y-step, width, height);*/

	int2 rb; rb.x = x+step; rb.y = y+step;
	int2 lt; lt.x = x-step; lt.y = y-step;
	int2 lb; lb.x = x-step; lb.y = y+step;
	int2 rt; rt.x = x+step; rt.y = y-step;

	curandState* localState = rngStates + threadIdx.y*blockDim.x + threadIdx.x;
	pixel[RED]		= createColor( surface, pitch, localState, lt, lb, rt, rb, width, height, RED );
	pixel[GREEN]	= createColor( surface, pitch, localState, lt, lb, rt, rb, width, height, GREEN );
	pixel[BLUE]		= createColor( surface, pitch, localState, lt, lb, rt, rb, width, height, BLUE );
	pixel[ALPHA]	= 1.0f;

}

__device__ void square( unsigned char *surface, size_t pitch, float* pixel,
	curandState* const rngStates, int x, int y, int step, int width, int height ){

	/*float* t = getPixel( surface, pitch, x, y+step, width, height );
	float* b = getPixel( surface, pitch, x, y-step, width, height );
	float* l = getPixel( surface, pitch, x-step, y, width, height );
	float* r = getPixel( surface, pitch, x+step, y, width, height );*/

	int2 t; t.x = x; t.y = y-step;
	int2 b; b.x = x; b.y = y+step;
	int2 l; l.x = x-step; l.y = y;
	int2 r; r.x = x+step; r.y = y;

	curandState* localState = rngStates + threadIdx.y*blockDim.x + threadIdx.x;
	/*pixel[RED]		= createColor( localState, step, t, b, l, r, RED );
	pixel[GREEN]	= createColor( localState, step, t, b, l, r, GREEN );
	pixel[BLUE]		= createColor( localState, step, t, b, l, r, BLUE );*/
	pixel[RED]		= createColor( surface, pitch, localState, t, b, l, r, width, height, RED );
	pixel[GREEN]	= createColor( surface, pitch, localState, t, b, l, r, width, height, GREEN );
	pixel[BLUE]		= createColor( surface, pitch, localState, t, b, l, r, width, height, BLUE );
	pixel[ALPHA]	= 1.0f;

	__syncthreads();
}

__global__ void cuke_diamondSquare( unsigned char *surface, int width,
	int height, size_t pitch, float t, int step,
	curandState* const rngStates )
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= width || y >= height) return;

	float* pixel = getPixel( surface, pitch, x, y, width, height );

	int x2 = x%(step);
	int y2 = y%(step);
	if( (x2 == 0) && (y2 == 0) ){
		//if( pixel[RED] < 1.1f ){
			

#pragma region oldWrappAlgs
			/*int lx = (x-step+width) % (width);
			int rx = (x+step+width) % (width);
			int ty = (y-step+height) % (height);
			int by = (y+step+height) % (height);*/
			
			/*int lx = (x-step+width-1) % (width-1);
			int rx = (x+step+width-1) % (width-1);
			int ty = (y-step+height-1) % (height-1);
			int by = (y+step+height-1) % (height-1);*/

			/*int lx = (x-step)<0				? (x-step)+(width-1)  : (x-step);
			int rx = (x+step)>(width-1)		? (x+step)-(width-1)  : (x+step);
			int ty = (y-step)<0				? (y-step)+(height-1) : (y-step);
			int by = (y+step)>(height-1)	? (y+step)-(height-1) : (y+step);*/
#pragma endregion

			bool xMid = (x+step) % (step*2) == 0;
			bool yMid = (y+step) % (step*2) == 0;

			if( xMid && yMid ){
				//diamond
				//pixel[GREEN]	= (step+1)/128.0f;
				//pixel[GREEN]	= 0.5f;
				//pixel[BLUE]		= (step+1)/32.0f;
				//pixel[ALPHA]	= 1.0f;
				diamond( surface, pitch, pixel, rngStates, x, y, step, width, height );
			}
			
			__syncthreads();
			
			if( xMid != yMid ){
				//square
				//pixel[GREEN]	= (step+1)/32.0f;
				//pixel[BLUE]		= (step+1)/128.0f;
				//pixel[BLUE]		= 0.5f;
				//pixel[ALPHA]	= 1.0f;
				square( surface, pitch, pixel, rngStates, x, y, step, width, height );
			}
			__syncthreads();
		//}
	}
	//pixel[RED]		= 0.2f;

}

__global__ void cuke_clear( unsigned char *surface, int width,
								   int height, size_t pitch )
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the pixel at (x,y)
	float* pixel = (float *)(surface + y*pitch) + 4*x;

	pixel[RED]		= 0.0f;
	pixel[GREEN]	= 0.0f;
	pixel[BLUE]		= 0.0f;
	pixel[ALPHA]	= 1.0f;

	float val = 0.5f;

	if( x==0 && y==0 ){
		pixel[RED]		= val;
		pixel[GREEN]	= val;
		pixel[BLUE]		= val;
	} else if( x==width-1 && y==0 ){
		pixel[RED]		= val;
		pixel[GREEN]	= val;
		pixel[BLUE]		= val;
	} else if( x==0 && y==height-1 ){
		pixel[RED]		= val;
		pixel[GREEN]	= val;
		pixel[BLUE]		= val;
	} else if( x==width-1 && y==height-1 ){
		pixel[RED]		= val;
		pixel[GREEN]	= val;
		pixel[BLUE]		= val;
	}

}

void cu_diamondSquare( void *surface, int width, int height, size_t pitch, 
					  float t )
{
	cudaError_t error = cudaSuccess;

	dim3 block = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 grid = dim3((width+block.x-1)/block.x, (height+block.y-1)/block.y);
	
	// init curand
	curandState *d_rngStates = NULL;
	cudaError_t cudaResult = cudaMalloc((void **)&d_rngStates, grid.x * block.x * sizeof(curandState));
	unsigned int seed = 1234;
	initRNG<<<grid, block>>>( d_rngStates, seed );

	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	// Do the magic
	cuke_clear<<<grid,block>>>( (unsigned char *)surface, width, height, pitch );
	gpuErrchk( cudaDeviceSynchronize() );


	/*static int k=1;
	k*=2;
	k = k>512 ? 64 : k;*/

	static int k=width-1;
	k/=2;
	k=0;
	for( int i=width; i>k; i/=2 ){
		cuke_diamondSquare<<<grid,block>>>( (unsigned char *)surface, width, height, pitch, (float)i, i, d_rngStates );
		gpuErrchk( cudaDeviceSynchronize() );
		//break;
	}

	k = k<1 ? width-1 : k;

	error = cudaGetLastError();

	if( error != cudaSuccess ){
		printf( "cuda_kernel_texture_2d() failed to launch error = %d\n",
			error );
	}

	// cleanup
	if (d_rngStates)
	{
		cudaFree(d_rngStates);
		d_rngStates = 0;
	}
}