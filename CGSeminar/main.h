#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <rapidxml-1.13/rapidxml.hpp>
#include <rapidxml-1.13/rapidxml_utils.hpp>

struct curve_info
{
	unsigned int * number_of_segments;
	unsigned int * curve_map;
	unsigned int* curve_index;
	
	float3* control_points;
	float3* bounding_boxes;


	unsigned int * number_of_colors_left;
	uint2* color_left_index;
	float3* color_left;
	float* color_left_u;

	unsigned int * number_of_colors_right;
	uint2* color_right_index;
	float3* color_right;
	float* color_right_u;

	float4* sample_accumulator;

	curandState_t* rand_state;

	uint2* image_size;
};

struct window_info
{
	curve_info* curve_pointers_device;

	uint2 window_size;
	uint2 mouse_pos;
	int window_type;

	unsigned int sample_count;
};

int main();
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
static void pushColor(rapidxml::xml_node<>* color_node, std::vector<uint2>& ind, std::vector<float>& color_u, std::vector<float3>& color);