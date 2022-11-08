#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <rapidxml-1.13/rapidxml.hpp>
#include <rapidxml-1.13/rapidxml_utils.hpp>





int main();
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
static void pushColor(rapidxml::xml_node<>* color_node, std::vector<uint2>& ind, std::vector<float>& color_u, std::vector<float3>& color);