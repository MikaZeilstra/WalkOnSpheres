
#pragma once
#include "kernel.cuh"
#include "vec_mult.cuh"

#define THREADS_PER_BLOCK 32
#define DISTANCE_INTIAL 10e5
#define LOCAL_MINIMUM_EPS 1e-2
#define NUM_LIN_MINIMUM_SCANS  10
#define NUM_BIN_MINIMUM_SCANS 100

CUstream_st* stream;

__global__ void set_initial_distance_kernel(float4* image, uint2* size)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < size->x && y < size->y) {
        image[x + y*size->y] = { DISTANCE_INTIAL,DISTANCE_INTIAL,DISTANCE_INTIAL, DISTANCE_INTIAL};
    }


}

__device__ __inline__ float3 get_bezier_point(float3* control_points, float t) {
    return  (1 - t) * (1 - t) * (1 - t) * control_points[0] +
        3 * (1 - t) * (1 - t) * t * control_points[1] +
        3 * (1 - t) * t * t * control_points[2] +
        t * t * t * control_points[3];
}

__device__ __inline__ float3 get_bezier_normal(float3* control_points, float t) {
    float3 tangent =
        -3 * (1 - t) * (1 - t) * control_points[0] +
        3 * (1 - t) * (1 - t) * control_points[1] - 6 * t * (1 - t) * control_points[1] -
        3 * t * 2 * control_points[2] +
        6 * t * (1 - t) * control_points[2] +
        3 * t * 2 * control_points[3];

    return { -tangent.y,tangent.x };
}

__device__ float find_local_minimum(float3* control_points,float3 point, float min_x, float max_x) {
    float m = min_x;
    float n = max_x;
    float k = 0;
    #pragma unroll
    while(abs(n-m) > LOCAL_MINIMUM_EPS) {
        k = m + (n - m) / 2;
        if (sqr_dist(get_bezier_point(control_points, k - LOCAL_MINIMUM_EPS),point) < sqr_dist(get_bezier_point(control_points, k + LOCAL_MINIMUM_EPS),point)) {
            n = k;
        } else {
            m = k;
        }
    }
    return k;
}

__device__ float find_closest_bezier_point(float3* control_points,float3 point) {
    float mindex = 0;
    float min_dist = 10e10;
    float c_dist = 0;
    float t = 0;
    #pragma unroll
    for (int i = 0; i <= NUM_LIN_MINIMUM_SCANS; i++) {
        t = i * (1.0f / NUM_LIN_MINIMUM_SCANS);
        c_dist = sqr_dist(get_bezier_point(control_points, t), point);
        if (c_dist < min_dist) {
            mindex = i;
            min_dist = c_dist;
        }
    }
    return find_local_minimum(control_points, point, fmaxf((mindex - 1) * (1.0f / NUM_LIN_MINIMUM_SCANS), 0), fminf((mindex + 1) * (1.0f / NUM_LIN_MINIMUM_SCANS), 1));
}


__global__ void make_distance_map_kernel(float4* image, uint2* size, curve_info* curve_pointers)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float3 point = { (float)x / size->x , (float)y / size->y };

    if (x < size->x && y < size->y) {
        // For each curve segment calculate the distance to this point
        for (int segment_index = 0; segment_index < *(curve_pointers->number_of_segments); segment_index++) {

            //Calculate distance to the bounding box of the curve
            float dx = fmaxf(abs(curve_pointers->bounding_boxes[segment_index * 2].x - point.x) - curve_pointers->bounding_boxes[(segment_index * 2) + 1].x / 2, 0);
            float dy = fmaxf(abs(curve_pointers->bounding_boxes[segment_index * 2].y - point.y) - curve_pointers->bounding_boxes[(segment_index * 2) + 1].y / 2, 0);
            float bb_distance = sqrtf(dx * dx + dy * dy);

            //If the distance to the bounding box is smaller than the current minimum find the actual distance to the segment
            if (bb_distance < image[x + y * size->y].x) {
                float closest_t = find_closest_bezier_point(&(curve_pointers->control_points[segment_index * 4]), point);
                float distance = sqrtf(sqr_dist(get_bezier_point(&(curve_pointers->control_points[segment_index * 4]), closest_t), point));
                
                //If the actual distance is also smaller update the distance and color map
                if (distance < image[x + y * size->y].x) {
                    image[x + y * size->y] = { distance,distance,distance,1 };
                }
                
            }

        }
    }


}




namespace KernelWrapper{
    void set_initial_distance(uint2 size, float4* image_device, uint2* size_device) {
        dim3 dim_threads_per_block = { THREADS_PER_BLOCK ,THREADS_PER_BLOCK, 1 };
        dim3 dim_block_grid = { ((unsigned int) ceil(size.x / THREADS_PER_BLOCK)), ((unsigned int) ceil(size.y / THREADS_PER_BLOCK)) ,1};

        set_initial_distance_kernel << <dim_block_grid, dim_threads_per_block,0,stream >> > (image_device,size_device);
    }

    void make_distance_map(uint2 size, float4* image_device, uint2* size_device, curve_info* curve_pointers) {
        dim3 dim_threads_per_block = { THREADS_PER_BLOCK ,THREADS_PER_BLOCK, 1 };
        dim3 dim_block_grid = { ((unsigned int)ceil(size.x / THREADS_PER_BLOCK)), ((unsigned int)ceil(size.y / THREADS_PER_BLOCK)) ,1 };

        make_distance_map_kernel << <dim_block_grid, dim_threads_per_block, 0, stream >> > (image_device, size_device, curve_pointers);
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