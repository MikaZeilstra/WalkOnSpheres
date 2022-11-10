
#pragma once
#include "kernel.cuh"
#include "vec_mult.cuh"

#define THREADS_PER_BLOCK 32
#define DISTANCE_INTIAL 10e5
#define LOCAL_MINIMUM_EPS 1e-4
#define NUM_LIN_MINIMUM_SCANS  20
#define NUM_BIN_MINIMUM_SCANS 100
#define WALKING_SPHERES_EPS 1e-3
#define WALKING_SPHERES_MAX_WALK 100
#define DISTANCE_MAP_EPS 5e-3

CUstream_st* stream;

__global__ void set_initial_distance_kernel(float4* image, uint2* size)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < size->x && y < size->y) {
        image[x + y*size->x] = { DISTANCE_INTIAL,DISTANCE_INTIAL,DISTANCE_INTIAL, DISTANCE_INTIAL};
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

    //Rotate tangent (cc) left and return it
    return { tangent.y,-tangent.x };
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
    float min_dist = 10e5;
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

__device__ float3 find_color(uint2& index, float t,float3* colors, float*color_t) {
    int i = index.x;

    
    while (i < index.x + index.y && color_t[i + 1] < t) {
        i++;
    }
    float r = (t - color_t[i]) / (color_t[i + 1] - color_t[i]);

    return {
        (colors[i].x * (1 - r)) + (colors[i+1].x * r)  ,
        (colors[i].z * (1 - r)) + (colors[i + 1].y * r),
        (colors[i].z * (1 - r)) + (colors[i + 1].z * r)
    };
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
            if (bb_distance < image[x + y * size->x].w) {
                float closest_t = find_closest_bezier_point(&(curve_pointers->control_points[segment_index * 4]), point);
                float3 curve_point = get_bezier_point(&(curve_pointers->control_points[segment_index * 4]), closest_t);
                float distance = sqrtf(sqr_dist(curve_point, point));
                                
                //If the actual distance is also smaller update the distance and color map
                if (distance < image[x + y * size->x].w) {
                    float3 curve_normal = get_bezier_normal(&(curve_pointers->control_points[segment_index * 4]), closest_t);
                    float3 color = { 0,0,0 };
                    unsigned int curve_index = curve_pointers->curve_map[segment_index];
                    float curve_t = closest_t + curve_pointers->curve_index[segment_index];
                    // find the color with the correct side
                    if (dot_prod(curve_normal, curve_point - point) > 0) {
                        color = find_color(curve_pointers->color_right_index[curve_index], curve_t, curve_pointers->color_right, curve_pointers->color_right_u);
                    }
                    else {
                       color = find_color(curve_pointers->color_left_index[curve_index], curve_t, curve_pointers->color_left, curve_pointers->color_left_u);
                    }


                    image[x + y * size->x] = { color.x,color.y,color.z,fmaxf(distance- DISTANCE_MAP_EPS,0)};
                }
                
            }

        }
    }


}

__device__ float4 interpolate_bilinear(float4* distance_map,uint2* size, float3& point){
    float x = point.x * size->x;
    float y = point.y * size->y;

    int2 p1 = { (int)floorf(x),(int)floorf(y) };
    int2 p2 = { (int)ceilf(x),(int)floorf(y) };
    int2 p3 = { (int)floorf(x),(int)ceilf(y) };
    int2 p4 = { (int)ceilf(x),(int)ceilf(y) };

    float x_r = point.x - p1.x;
    float y_r = point.y - p1.y;

    float4 v1 = (1 - x_r) * distance_map[p1.x + p1.y * size->x] + x_r * distance_map[p2.x + p2.y * size->x];
    float4 v2 = (1 - x_r) * distance_map[p3.x + p3.y * size->x] + x_r * distance_map[p4.x + p4.y * size->x];

    float4 v3 = (1 - y_r) * v1 + (y_r)*v2;
    //v3.w = fminf(fminf(distance_map[p1.x + p1.y * size->x].w, distance_map[p2.x + p2.y * size->x].w), fminf(distance_map[p3.x + p3.y * size->x].w, distance_map[p4.x + p4.y * size->x].w));

    return distance_map[(int) round(x) + (int) round(y) * size->x];

}

__global__ void sample_kernel(float4* image, uint2* size, float4* distance_map, curandState_t* rand_states,unsigned int* sample_count) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float3 point = { (float)x / size->x , (float)y / size->y };
    float4 value = distance_map[x + y * size->x];
    float rot_cos = 0;
    float rot_sin = 1;
    int i = 0;
    if (x < size->x && y < size->y) {
        while(value.w > WALKING_SPHERES_EPS && i < WALKING_SPHERES_MAX_WALK) {
            sincospif(curand_uniform(&rand_states[x+y*size->x]), &rot_sin, &rot_cos);
            point = { fminf(fmaxf(point.x + rot_cos * value.w,0),1.0f), fminf(fmaxf(point.y + rot_sin * value.w,0),1.0f),0 };
            //value = interpolate_bilinear(distance_map, size, point);
            value = distance_map[min(max((int)round(point.x * size->x), 0), size->x) +  min(max((int)round(point.y * size->y), 0), (size->y) - 1) * size->x];
            i++;
        }
    }
    float4 final_v = { value.x, value.y,value.z, 1};
    image[x + y * size->x] = image[x + y * size->x] + (1.0f / ( * sample_count)) * final_v;
}

__global__ void setup_curand_kernel(curandState_t* states, uint2* size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < size->x && y < size->y) {
        curand_init(x + y * size->x, size->x, size->y, &states[x + y * size->x]);
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

    void sample(uint2 size,float4* image, uint2* size_device , float4* distance_map, curandState_t* rand_states, unsigned int* sample_count) {
        dim3 dim_threads_per_block = { THREADS_PER_BLOCK ,THREADS_PER_BLOCK, 1 };
        dim3 dim_block_grid = { ((unsigned int)ceil(size.x / THREADS_PER_BLOCK)), ((unsigned int)ceil(size.y / THREADS_PER_BLOCK)) ,1 };

        sample_kernel << <dim_block_grid, dim_threads_per_block, 0, stream >> > (image,size_device,distance_map,rand_states,sample_count);
    }

    void setup_curand(uint2 size, curandState_t* states, uint2* size_device) {
        dim3 dim_threads_per_block = { THREADS_PER_BLOCK ,THREADS_PER_BLOCK, 1 };
        dim3 dim_block_grid = { ((unsigned int)ceil(size.x / THREADS_PER_BLOCK)), ((unsigned int)ceil(size.y / THREADS_PER_BLOCK)) ,1 };

        setup_curand_kernel << <dim_block_grid, dim_threads_per_block, 0, stream >> > (states, size_device);
    }

}

void GPU_setup() {
    cudaFree(0);
    CALL_CHECK(cudaStreamCreate(&stream));
}

void* GPU_malloc(size_t size) {
    void* device_ptr = nullptr;
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&device_ptr), size, stream));
    return device_ptr;
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