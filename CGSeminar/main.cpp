#pragma once
#define GLFW_INCLUDE_NONE

#define USE_DIFFUSION_CURVE_SAVE true

#define _USE_MATH_DEFINES

#include "kernel.cuh"
#include "main.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include <cmath>

#include <stdio.h>
#include <iostream>
#include <vector>
#include <filesystem>

#include <rapidxml-1.13/rapidxml.hpp>
#include <rapidxml-1.13/rapidxml_utils.hpp>

const std::string curve_file = "/xmls/arch.xml";
float initial_distance_value = 10e5;

char window_name[30];

int main() {

	//Setup cuda
	GPU_setup();
	
	//Setup rand
	srand(time(NULL));

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


	//Initialize counters
	int current_segment = 0;
	int current_curve = 0;
	int current_curve_segment = 0;

	unsigned int n_colors_left = 0;
	unsigned int n_colors_right = 0;
	unsigned int n_segments = 0;

	//make temp vars for bounding boxes
	float3 bb_min ={};
	float3 bb_max ={};

	//Read Diffusion curves
	for (rapidxml::xml_node<>* curve = curve_set->first_node(); curve; curve = curve->next_sibling()) {
		
		
		
		current_curve_segment = 0;
		set_node = curve->first_node("control_points_set", 18);

		current_node = set_node->first_node();

		//Loop over all splines
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
				if (USE_DIFFUSION_CURVE_SAVE) {
					vertices.back().y = 1 - vertices.back().y;
				}

				//update spline bounding box as simply max/min of x/y
				bb_min = { std::min(vertices[vertices.size() - 1].x ,bb_min.x) ,std::min(vertices[vertices.size() - 1].y ,bb_min.y),0 };
				bb_max = { std::max(vertices[vertices.size() - 1].x,bb_max.x) ,std::max(vertices[vertices.size() - 1].y ,bb_max.y),0 };
				current_node = current_node->next_sibling();
			}

			//Repeat for last control point
			vertices.push_back({
				(float)std::atof((current_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "y" : "x", 1))->value()) /image_size.x,
				(float)std::atof((current_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "x" : "y", 1))->value()) / image_size.y,
				0
			});

			if (USE_DIFFUSION_CURVE_SAVE) {
				vertices.back().y = 1 - vertices.back().y;
			}
			bb_min = { std::min(vertices[vertices.size() - 1].x ,bb_min.x) ,std::min(vertices[vertices.size() - 1].y ,bb_min.y),0 };
			bb_max = { std::max(vertices[vertices.size() - 1].x,bb_max.x) ,std::max(vertices[vertices.size() - 1].y ,bb_max.y),0 };

			//Update the host variables to the current state
			current_segment += 4;
			float3 bb_dimensions = { bb_max.x - bb_min.x, bb_max.y - bb_min.y ,0 };

			//Push the gathered info in the buffers
			bounding_boxes.push_back({bb_dimensions.x / 2 + bb_min.x, bb_dimensions.y / 2 + bb_min.y,0 });
			bounding_boxes.push_back(bb_dimensions);
			curve_map.push_back(current_curve);
			curve_index.push_back(current_curve_segment++);
		}

		//Set the current node to the left color nodes
		set_node = curve->first_node( "left_colors_set" , 15);
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

	//Setup GLFW
	glfwInit();
	GLFWwindow* window = glfwCreateWindow(image_size.x, image_size.y, "LearnOpenGL", NULL, NULL);
	
	//Set our own info object to the window user pointer for acces int the callback functions
	window_info info;
	glfwSetWindowUserPointer(window, &info);
	info.window_size = image_size;

	//set the callbacks
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetKeyCallback(window, key_callback);

	//Intialize window name to sample
	snprintf(window_name, 20, "Sample count : %u", info.sample_count);
	glfwSetWindowTitle(window, window_name);
	
	//Check if initialization was correct
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

	//Setup viewport after loading glad
	glViewport(0, 0, image_size.x, image_size.y);
	
	//Turn off vsync
	glfwSwapInterval(0);

	//Setup opengl
	gladLoadGL();
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);
	
	//Setup initial screen
	glfwSwapBuffers(window);

	//Create PBO for output
	//PBO opject
	GLuint final_pbo;


	glGenBuffers(1, &final_pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, final_pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * sizeof(GLfloat) * image_size.x * image_size.y, NULL, GL_DYNAMIC_DRAW);


	cudaGraphicsResource* pbo_final_resource;
	cudaGraphicsGLRegisterBuffer(&pbo_final_resource, final_pbo, cudaGraphicsRegisterFlagsWriteDiscard);

	float4* pbo_final_device;
	


	//Upload data to GPU
	curve_info device_pointers;

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


	//Alocate memory for the maps and accumulator
	device_pointers.distance_map = reinterpret_cast<float4*>(GPU_malloc(sizeof(float4) * image_size.x * image_size.y));
	device_pointers.image_table[0] = device_pointers.distance_map;
	device_pointers.boundary_conditions = reinterpret_cast<float4*>(GPU_malloc(sizeof(float4) * image_size.x * image_size.y));
	device_pointers.image_table[1] = device_pointers.boundary_conditions;
	device_pointers.current_solution = reinterpret_cast<float4*>(GPU_malloc(sizeof(float4) * image_size.x * image_size.y));
	device_pointers.image_table[2] = device_pointers.current_solution;
	device_pointers.sample_accumulator = reinterpret_cast<float4*>(GPU_malloc(sizeof(float4) * image_size.x * image_size.y));
	
	//Upload image size
	device_pointers.image_size = reinterpret_cast<uint2*>(GPU_upload(sizeof(uint2), &image_size));
	
	//Reserve space for curand_states and create them
	device_pointers.rand_state = reinterpret_cast<curandState_t*>(GPU_malloc(sizeof(curandState_t) * image_size.x * image_size.y));
	KernelWrapper::setup_curand(image_size, device_pointers.rand_state, device_pointers.image_size);
	
	//Upload all the pointer container to the gpu 
	curve_info* curve_info_device = reinterpret_cast<curve_info*>(GPU_upload(sizeof(curve_info), &device_pointers));
	info.curve_pointers_device = curve_info_device;

	//Create distance and boundary values map
	KernelWrapper::set_initial_distance(image_size, device_pointers.distance_map, device_pointers.image_size);
	KernelWrapper::make_distance_map(image_size, device_pointers.distance_map, device_pointers.boundary_conditions, curve_info_device);

	//Dowload the distance map back to the cpu for use in drawing circles
	float4* distance_map = reinterpret_cast<float4*>(malloc(sizeof(float4) * image_size.x * image_size.y));
	GPU_download(sizeof(float4) * image_size.x * image_size.y, device_pointers.distance_map, distance_map);

	//Make sure all the queued work is done
	GPU_sync();

	//Render loop
	while (!glfwWindowShouldClose(window))
	{
		//Get pointer to the PBO
		cudaGraphicsMapResources(1, &pbo_final_resource, NULL);
		cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pbo_final_device), NULL, pbo_final_resource);
		

		//Copy the correct window to the PBO if it has changed
		if (info.switch_window) {
			GPU_copy(sizeof(float4) * image_size.x * image_size.y, device_pointers.image_table[info.window_type], pbo_final_device);

			info.switch_window = false;
		}

		//If sample taking is unpaused or we want 1 sample sample and copy it to PBO
		if (!info.pause || info.next_sample) {

			KernelWrapper::sample(image_size, curve_info_device, info.sample_count);
			GPU_copy(sizeof(float4) * image_size.x * image_size.y, device_pointers.image_table[info.window_type], pbo_final_device);

			info.sample_count++;

			//Reset wheter to take 1 sample
			info.next_sample = false;
		}
		//Draw circle if needed
		else if (info.draw_circle) {
			GPU_copy(sizeof(float4) * image_size.x * image_size.y, device_pointers.image_table[info.window_type], pbo_final_device);

			//Set intial positions and invert y (we assume 0,0 in bottom left)
			int x = fminf(fmaxf(round(info.mouse_pos.x),0), info.window_size.x);
			int y = image_size.y - fminf(fmaxf(round(info.mouse_pos.y), 0), info.window_size.y);

			//Initialize temporary vars
			float2 point = {x/(float)info.window_size.x ,  y / (float) info.window_size.y}; //starting point of the walk
			float distance = distance_map[x + y * image_size.x].x; //Distanc to closest point on curve
			float rot_cos = 0; //random cos value
			float rot_sin = 1; //random sin value
			int i = 0;

			//Draw circle from initial point
			KernelWrapper::create_circle(image_size, curve_info_device, pbo_final_device, point, distance);


			//Keep walking untill we are close enough or we have hit the max
			while (distance > WALKING_SPHERES_EPS && i < WALKING_SPHERES_MAX_WALK) {
				//Take random number between 0 and 1
				float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				
				//Take cos and sin for random point on circle
				rot_cos = cos(2 * r * M_PI);
				rot_sin = sin(2 * r * M_PI);

				//Take step to the new point on circle with radius distance to the closest point centered on the current point
				point = { fminf(fmaxf(point.x + rot_cos * distance,0),1.0f), fminf(fmaxf(point.y + rot_sin * distance,0),1.0f)};

				//Find the new distance to the closest point
				distance = distance_map[(int) round(fminf(fmaxf((int)round(point.x * image_size.x), 0), image_size.x) + fminf(fmaxf((int)round(point.y * image_size.y), 0), (image_size.y) - 1) * image_size.x)].w;

				//Draw the circle on the new point
				KernelWrapper::create_circle(image_size, curve_info_device, pbo_final_device, point, distance);

				//Update walk counter
				i++;


			}

			//Reset wheter to draw circle
			info.draw_circle = false;
		}

		//Make sure all queued work is done
		GPU_sync();

		//Return buffer control to OpenGL
		cudaGraphicsUnmapResources(1, &pbo_final_resource, NULL);

		//Blit buffer to framebuffer
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, final_pbo);
		glDrawPixels(image_size.x, image_size.y, GL_RGBA, GL_FLOAT, 0);

		//Show buffer and check for events
		glfwSwapBuffers(window);
		glfwPollEvents();
		
		//Set the window name to current sample count
		snprintf(window_name, 30, "Sample count : %u", info.sample_count);
		glfwSetWindowTitle(window, window_name );
	}
	
	


	glfwTerminate();
	return 0;
}

//Helper functionf for reading colors from diffusion curve file
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


//Callback functions
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	void* data = glfwGetWindowUserPointer(window);
	window_info* info = static_cast<window_info*>(data);

	//Cycle windows with arrow keys
	if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS) {
		info->window_type = (info->window_type + 1) % 3;
		info->switch_window = true;
	}
	else if (key == GLFW_KEY_LEFT && action == GLFW_PRESS) {
		info->window_type = ((info->window_type - 1)+3) % 3;
		info->switch_window = true;
	}
	//Reset samples with R
	else if (key == GLFW_KEY_R && action == GLFW_PRESS) {
		KernelWrapper::reset_samples(info->window_size, info->curve_pointers_device);
		info->sample_count = 1;
	}
	//Pause and unpause with space
	else if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
		info->pause = !info->pause;
	}
	//Take 1 sample with D
	else if (key == GLFW_KEY_D && action == GLFW_PRESS) {
		info->next_sample = true;
	}
	//Draw circle with C
	else if (key == GLFW_KEY_C && action == GLFW_PRESS) {
		glfwGetCursorPos(window, &(info->mouse_pos.x), &(info->mouse_pos.y));
		info->draw_circle = true;
	}

}