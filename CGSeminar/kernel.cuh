#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#ifndef CALL_CHECK
#define CALL_CHECK(call) \
    if(call != 0){          \
        std::cerr << "Error in Optix/CUDA call with number " << call << " at line " << __LINE__ << " in file " << __FILE__ << std::endl; \
        throw std::exception(); \
    }
#endif // !

namespace KernelWrapper {
	void set_background(uint2 size, float4* image_device, uint2* size_device);
}

void GPU_setup();
void* GPU_upload(size_t size, void* data);
void GPU_download(size_t size, void* data_device, void* dst);
void GPU_free(void* device_ptr);
void GPU_sync();