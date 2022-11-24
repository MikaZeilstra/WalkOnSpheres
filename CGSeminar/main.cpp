#pragma once
#define GLFW_INCLUDE_NONE

#define USE_DIFFUSION_CURVE_SAVE true

#include "kernel.cuh"
#include "main.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include <stdio.h>
#include <iostream>
#include <vector>
#include <filesystem>

#include <rapidxml-1.13/rapidxml.hpp>
#include <rapidxml-1.13/rapidxml_utils.hpp>

const std::string curve_file = "/arch.xml";
float initial_distance_value = 10e5;

char window_name[20];

int main() {

	GPU_setup();

	//Load the xml file
	rapidxml::file<> xmlFile((std::filesystem::current_path().generic_string() + curve_file).c_str());
	rapidxml::xml_document<> doc;
	doc.parse<0>(xmlFile.data());
	rapidxml::xml_node<>* curve_set = doc.first_node();
	rapidxml::xml_node<>* set_node;
	rapidxml::xml_node<>* current_node;

	uint2 image_size = { std::atoi(curve_set->first_attribute("image_width")->value()) ,std::atoi(curve_set->first_attribute("image_height")->value()) };

	//Control point info
	std::vector<float3> vertices = {};						//The control points for all curves
	std::vector<float3> bounding_boxes = {};				//Bounding boxes as [i] = {center x , center y}   and  [i+1] = {width,height} 
	std::vector<unsigned int> curve_map = {};				//Curve number for each segment
	std::vector<unsigned int> curve_index = {};				//Segment number in each curve per segment

	//Color info
	std::vector<uint2> color_left_index{};					//for each curve starting index (x) and color control point count (y) in other arrays
	std::vector<float3> color_left = {};					//Color for each color_control point
	std::vector<float> color_left_u = {};					//curve parameter for each color control point

	//Same as left but for right
	std::vector<uint2> color_right_index{};
	std::vector<float3> color_right = {};
	std::vector<float> color_right_u = {};



	int current_segment = 0;
	int current_curve = 0;
	int current_curve_segment = 0;

	unsigned int n_colors_left = 0;
	unsigned int n_colors_right = 0;
	unsigned int n_segments = 0;

	float3 bb_min ={};
	float3 bb_max ={};

	curve_info device_pointers;
	

	for (rapidxml::xml_node<>* curve = curve_set->first_node(); curve; curve = curve->next_sibling()) {
		
		
		
		current_curve_segment = 0;
		set_node = curve->first_node("control_points_set", 18);

		current_node = set_node->first_node();

		while (current_node->next_sibling()) {
			bb_min = { 10e10,10e10 ,0};
			bb_max = { -10e10,-10e10,0 };
			//Insert the 4 vertexes of the spline
			for (int i = 0; i < 3; i++) {
				vertices.push_back({
					(float)std::atof((current_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "y" : "x", 1))->value())/image_size.x,
					(float)std::atof((current_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "x" : "y", 1))->value()) / image_size.y,
					0
				});
				bb_min = { std::min(vertices[vertices.size() - 1].x ,bb_min.x) ,std::min(vertices[vertices.size() - 1].y ,bb_min.y),0 };
				bb_max = { std::max(vertices[vertices.size() - 1].x,bb_max.x) ,std::max(vertices[vertices.size() - 1].y ,bb_max.y),0 };
				current_node = current_node->next_sibling();
			}

			vertices.push_back({
				(float)std::atof((current_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "y" : "x", 1))->value()) /image_size.x,
				(float)std::atof((current_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "x" : "y", 1))->value()) / image_size.y,
				0
			});
			bb_min = { std::min(vertices[vertices.size() - 1].x ,bb_min.x) ,std::min(vertices[vertices.size() - 1].y ,bb_min.y),0 };
			bb_max = { std::max(vertices[vertices.size() - 1].x,bb_max.x) ,std::max(vertices[vertices.size() - 1].y ,bb_max.y),0 };

			//Update the host variables to the current state
			current_segment += 4;
			float3 bb_dimensions = { bb_max.x - bb_min.x, bb_max.y - bb_min.y ,0 };

			bounding_boxes.push_back({bb_dimensions.x / 2 + bb_min.x, bb_dimensions.y / 2 + bb_min.y,0 });
			bounding_boxes.push_back(bb_dimensions);
			curve_map.push_back(current_curve);
			curve_index.push_back(current_curve_segment++);
		}

		//Set the current node to the left color nodes
		set_node = curve->first_node("left_colors_set", 15);
		color_left_index.push_back({ n_colors_left ,0 });
		current_node = set_node->first_node();


		//Read all the left colors
		while (current_node) {
			pushColor(current_node, color_left_index, color_left_u, color_left);
			current_node = current_node->next_sibling();
		}


		//set the current node to the right color nodes
		set_node = curve->first_node("right_colors_set", 16);
		color_right_index.push_back({ n_colors_right ,0 });
		current_node = set_node->first_node();

		//Read all the right colors
		while (current_node) {
			pushColor(current_node, color_right_index, color_right_u, color_right);
			current_node = current_node->next_sibling();
		}

		//Make sure there is a color value for the last parameter value of the curve if we are using a diffusion curve save
		if (USE_DIFFUSION_CURVE_SAVE) {
			color_right.push_back(color_right.back());
			color_right_index.back().y++;
			color_right_u.push_back(current_curve_segment);

			color_left.push_back(color_left.back());
			color_left_index.back().y++;
			color_left_u.push_back(current_curve_segment);
		}

		n_colors_left += color_left_index.back().y;
		n_colors_right += color_right_index.back().y;

		current_curve++;
		n_segments += current_curve_segment;

	
	}

	glfwInit();
	GLFWwindow* window = glfwCreateWindow(image_size.x, image_size.y, "LearnOpenGL", NULL, NULL);
	window_info info;
	glfwSetWindowUserPointer(window, &info);
	info.window_size = image_size;

	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetKeyCallback(window, key_callback);

	snprintf(window_name, 20, "Sample count : %u", info.sample_count);
	glfwSetWindowTitle(window, window_name);
	
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	glViewport(0, 0, image_size.x, image_size.y);
	
	//Turn off vsync
	glfwSwapInterval(0);

	gladLoadGL();
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);
	
	glfwSwapBuffers(window);

	//Create PBO for output
	//PBO opject
	GLuint distance_pbo;
	GLuint color_pbo;
	GLuint final_pbo;

	glGenBuffers(1, &distance_pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, distance_pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * sizeof(GLfloat) * image_size.x * image_size.y, NULL, GL_DYNAMIC_DRAW);

	glGenBuffers(1, &color_pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, color_pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * sizeof(GLfloat) * image_size.x * image_size.y, NULL, GL_DYNAMIC_DRAW);

	glGenBuffers(1, &final_pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, final_pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * sizeof(GLfloat) * image_size.x * image_size.y, NULL, GL_DYNAMIC_DRAW);


	//Register PBO in CUDA for further usage
	cudaGraphicsResource* pbo_distance_resource;
	cudaGraphicsGLRegisterBuffer(&pbo_distance_resource, distance_pbo, cudaGraphicsRegisterFlagsWriteDiscard);

	cudaGraphicsResource* pbo_color_resource;
	cudaGraphicsGLRegisterBuffer(&pbo_color_resource, color_pbo, cudaGraphicsRegisterFlagsWriteDiscard);

	cudaGraphicsResource* pbo_final_resource;
	cudaGraphicsGLRegisterBuffer(&pbo_final_resource, final_pbo, cudaGraphicsRegisterFlagsWriteDiscard);


	//setup params.image as PBO
	
	float4* pbo_distance_device;
	float4* pbo_color_device;
	float4* pbo_final_device;
	


	//Upload data to GPU
	device_pointers.control_points = reinterpret_cast<float3*>(GPU_upload(sizeof(float3) * vertices.size(), vertices.data()));
	device_pointers.bounding_boxes = reinterpret_cast<float3*>(GPU_upload(sizeof(float3) * bounding_boxes.size(), bounding_boxes.data()));
	device_pointers.curve_map = reinterpret_cast<unsigned int*>(GPU_upload(sizeof(unsigned int) * curve_map.size(),curve_map.data()));
	device_pointers.curve_index = reinterpret_cast<unsigned int*>(GPU_upload(sizeof(unsigned int) * curve_index.size(),curve_index.data()));
	

	device_pointers.color_left_index = reinterpret_cast<uint2*>(GPU_upload(sizeof(uint2) * color_left_index.size(), color_left_index.data()));
	device_pointers.color_left = reinterpret_cast<float3*>(GPU_upload(sizeof(float3) * color_left.size(), color_left.data()));
	device_pointers.color_left_u = reinterpret_cast<float*>(GPU_upload(sizeof(float) * color_left_u.size(), color_left_u.data()));

	device_pointers.color_right_index = reinterpret_cast<uint2*>(GPU_upload(sizeof(uint2) * color_right_index.size(), color_right_index.data()));
	device_pointers.color_right = reinterpret_cast<float3*>(GPU_upload(sizeof(float3) * color_right.size(), color_right.data()));
	device_pointers.color_right_u = reinterpret_cast<float*>(GPU_upload(sizeof(float) * color_right_u.size(), color_right_u.data()));

	device_pointers.number_of_segments = reinterpret_cast<unsigned int*>(GPU_upload(sizeof(unsigned int), &n_segments));
	device_pointers.number_of_colors_left = reinterpret_cast<unsigned int*>(GPU_upload(sizeof(unsigned int), &n_colors_left));
	device_pointers.number_of_colors_right = reinterpret_cast<unsigned int*>(GPU_upload(sizeof(unsigned int), &n_colors_right));

	device_pointers.sample_accumulator = reinterpret_cast<float4*>(GPU_malloc(sizeof(float4) * image_size.x * image_size.y));
	

	device_pointers.image_size = reinterpret_cast<uint2*>(GPU_upload(sizeof(uint2), &image_size));
	
	//Reserve space for curand_states and create them
	device_pointers.rand_state = reinterpret_cast<curandState_t*>(GPU_malloc(sizeof(curandState_t) * image_size.x * image_size.y));
	KernelWrapper::setup_curand(image_size, device_pointers.rand_state, device_pointers.image_size);
	curve_info* curve_info_device = reinterpret_cast<curve_info*>(GPU_upload(sizeof(curve_info), &device_pointers));
	info.curve_pointers_device = curve_info_device;

	cudaGraphicsMapResources(1, &pbo_distance_resource, NULL);
	cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pbo_distance_device), NULL, pbo_distance_resource);

	cudaGraphicsMapResources(1, &pbo_color_resource, NULL);
	cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pbo_color_device), NULL, pbo_color_resource);

	KernelWrapper::set_initial_distance(image_size, pbo_distance_device,device_pointers.image_size);
	KernelWrapper::make_distance_map(image_size, pbo_distance_device, pbo_color_device, curve_info_device);

	cudaGraphicsMapResources(1, &pbo_final_resource, NULL);
	cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pbo_final_device), NULL, pbo_final_resource);

	
	float4* image = reinterpret_cast<float4*>(malloc(sizeof(float4) * image_size.x * image_size.y));
	float4* distance = reinterpret_cast<float4*>(malloc(sizeof(float4) * image_size.x * image_size.y));

	GPU_sync();
	cudaGraphicsUnmapResources(1, &pbo_distance_resource, NULL);
	cudaGraphicsUnmapResources(1, &pbo_color_resource, NULL);
	cudaGraphicsUnmapResources(1, &pbo_final_resource, NULL);

	info.sample_count = 1;

	while (!glfwWindowShouldClose(window))
	{
		cudaGraphicsMapResources(1, &pbo_color_resource, NULL);
		cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pbo_color_device), NULL, pbo_color_resource);


		cudaGraphicsMapResources(1, &pbo_final_resource, NULL);
		cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pbo_final_device), NULL, pbo_final_resource);

		KernelWrapper::sample(image_size, pbo_final_device, curve_info_device, pbo_color_device, info.sample_count);

		GPU_sync();

		cudaGraphicsUnmapResources(1, &pbo_color_resource, NULL);
		cudaGraphicsUnmapResources(1, &pbo_final_resource, NULL);

		if (info.window_type == 0) {
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, distance_pbo);
		}
		else if (info.window_type == 1) {
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, color_pbo);
		}
		else if (info.window_type == 2) {
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, final_pbo);
		}
			
		glDrawPixels(image_size.x, image_size.y, GL_RGBA, GL_FLOAT, 0);


		glfwSwapBuffers(window);
		glfwPollEvents();
		snprintf(window_name, 20, "Sample count : %u", info.sample_count);
		glfwSetWindowTitle(window, window_name );
		info.sample_count++;
	}
	
	


	glfwTerminate();
	return 0;
}

static void pushColor(rapidxml::xml_node<>* color_node, std::vector<uint2>& ind, std::vector<float>& color_u, std::vector<float3>& color) {
	float u = (std::atof(color_node->first_attribute("globalID", 8)->value()) / 10.0f);
	color.push_back({
		std::atoi(color_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "B" : "R",1)->value()) / 255.0f,
		std::atoi(color_node->first_attribute("G",1)->value()) / 255.0f,
		std::atoi(color_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "R" : "B",1)->value()) / 255.0f
		});
	color_u.push_back(u);
	ind.back().y++;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	void* data = glfwGetWindowUserPointer(window);
	window_info* info = static_cast<window_info*>(data);
	if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS) {
		info->window_type = (info->window_type + 1) % 3;
	}
	else if (key == GLFW_KEY_LEFT && action == GLFW_PRESS) {
		info->window_type = (info->window_type - 1) % 3;
	}
	else if (key == GLFW_KEY_R && action == GLFW_PRESS) {
		KernelWrapper::reset_samples(info->window_size, info->curve_pointers_device);
		info->sample_count = 0;
	}

}