#ifndef KERNEL_CUH
#define KERNEL_CUH
#include <cuda_runtime.h>

extern "C"
{
	int add();
	void cuda_texture_2d(void *surface, int width, int height, size_t pitch, float t);
}

#endif