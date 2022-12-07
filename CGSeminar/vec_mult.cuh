#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ __inline__ float3 operator*(const float& a, const float3& b) {

	return make_float3(a * b.x, a * b.y, a * b.z);

}

__device__ __inline__ float4 operator*(const float& a, const float4& b) {

	return make_float4(a * b.x, a * b.y, a * b.z,a * b.w);

}

__device__ __inline__ void operator+=(float4& a, const float4& b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}

__device__ __inline__ float3 operator+(const float3& a, const float3& b) {

	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);

}

__device__ __inline__ float4 operator+(const float4& a, const float4& b) {

	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);

}

__device__ __inline__ float3 operator-(const float3& a, const float3& b) {

	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);

}

__device__ __inline__ float sqr_dist(const float3& a, const float3& b) {

	return (a.x-b.x)* (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);

}

__device__ __inline__ float dot_prod(const float3& a, const float3& b) {

	return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);

}


__device__ __inline__ float dot_prod(const float2& a, const float2& b) {

	return (a.x * b.x) + (a.y * b.y);

}

__device__ __inline__ float norm(const float3& a) {

	return a.x * a.x +  a.y  * a.y + a.z * a.z;

}

