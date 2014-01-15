#pragma once
#include <cuda_runtime.h>

extern "C" {
	void cu_diamondSquare( void *surface, int width, int height,
		size_t pitch, float t );
}