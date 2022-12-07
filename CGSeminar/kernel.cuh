#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <curand_kernel.h>

#include "main.h"

#define THREADS_PER_BLOCK 32
#define DISTANCE_INTIAL 10e5
#define LOCAL_MINIMUM_EPS 1e-5
#define NUM_LIN_MINIMUM_SCANS  20
#define NUM_BIN_MINIMUM_SCANS 100
#define WALKING_SPHERES_EPS 1e-3
#define WALKING_SPHERES_MAX_WALK 100
#define DISTANCE_MAP_EPS 5e-3

#define CIRCLE_WIDTH 1e-3
#define CIRCLE_COLOR 0,0,0,1
#define CIRCLE_CENTER_COLOR 0,1,0,1


#ifndef CALL_CHECK
#define CALL_CHECK(call) \
    if(call != 0){          \
        std::cerr << "Error in Optix/CUDA call with number " << call << " at line " << __LINE__ << " in file " << __FILE__ << std::endl; \
        throw std::exception(); \
    }
#endif // !

namespace KernelWrapper {
	void set_initial_distance(uint2 size, float4* image_device, uint2* size_device);

    void reset_samples(uint2 size, curve_info* curve_info_device);

    void make_distance_map(uint2 size, float4* distance_device, float4* color_device, curve_info* curve_pointers);

    void sample(uint2 size, curve_info* curve_info_device, unsigned int sample_count);

    void setup_curand(uint2 size, curandState_t* states, uint2* size_device);

    void create_circle(uint2 size, curve_info* curve_info_device, float4* image, float2 circle_center, float radius);  
}

void GPU_setup();
void* GPU_malloc(size_t size);
void* GPU_upload(size_t size, void* data);
void GPU_download(size_t size, void* data_device, void* dst);
void GPU_free(void* device_ptr);
void GPU_sync();
void GPU_copy(rsize_t size, void* data_src, void* data_dst);
void GPU_start_timer();
float GPU_stop_timer(std::string description, bool use_newline=true);