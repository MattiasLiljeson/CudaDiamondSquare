#pragma once

#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

//
// Below stuph is shamelessly stolen from helper_cuda.h
//

template< typename T >
bool check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        /*
                std::stringstream ss;
                std::string msg("CUDA error at ");
                msg += file;
                msg += ":";
                ss << line;
                msg += ss.str();
                msg += " code=";
                ss << static_cast<unsigned int>(result);
                msg += ss.str();
                msg += " (";
                msg += _cudaGetErrorEnum(result);
                msg += ") \"";
                msg += func;
                msg += "\"";
                //throw msg;
                std::cerr  << msg <<"\n";
        */
        return true;
    }
    else
    {
        return false;
    }
}

#ifdef __DRIVER_TYPES_H__
// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
			file, line, errorMessage, (int)err, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
#endif