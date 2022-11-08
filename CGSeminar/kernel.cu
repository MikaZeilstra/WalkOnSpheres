
#pragma once
#include "kernel.cuh"

#define THREADS_PER_BLOCK 32 

CUstream_st* stream;

__global__ void set_background_kernel(float4* image, uint2* size)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < size->x && y < size->y) {
        image[x + y*size->y] = {1,0,0,1};
    }


}

namespace KernelWrapper{
    void set_background(uint2 size, float4* image_device, uint2* size_device) {
        dim3 dim_threads_per_block = { THREADS_PER_BLOCK ,THREADS_PER_BLOCK, 1 };
        dim3 dim_block_grid = { ((unsigned int) ceil(size.x / THREADS_PER_BLOCK)), ((unsigned int) ceil(size.y / THREADS_PER_BLOCK)) ,1};




        set_background_kernel << <dim_block_grid, dim_threads_per_block,0,stream >> > (image_device,size_device);
    }

}

void GPU_setup() {
    cudaFree(0);

    CALL_CHECK(cudaStreamCreate(&stream));
}

void* GPU_upload(size_t size, void* data) {
    void* device_ptr = nullptr;
    CALL_CHECK(cudaMallocAsync(&device_ptr, size, stream));
    CALL_CHECK(cudaMemcpyAsync(
        device_ptr,
        data,
        size,
        cudaMemcpyHostToDevice,
        stream
    ));
    return device_ptr;
}

void GPU_download(size_t size, void* data_device, void* dst) {
    CALL_CHECK(cudaMemcpyAsync(
        dst,
        data_device,
        size,
        cudaMemcpyDeviceToHost,
        stream
    ));
}


void GPU_free(void* device_ptr) {
    cudaFreeAsync(device_ptr, stream);
}

void GPU_sync() {
    CALL_CHECK(cudaDeviceSynchronize());
    CALL_CHECK(cudaStreamSynchronize(stream));
}