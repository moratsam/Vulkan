#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <optional>
#include <set>
#include <cstdint>
#include <algorithm>
#include <fstream>
#include <glm/glm.hpp>
#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "vulkanbase.h"

//------<CONSTS & MACROS>------
#define COLOR_FORMAT VK_FORMAT_R8G8B8A8_SRGB

const std::string DUMP_PATH = "./dump/";

float globalTime = 0.0f;
//------</CONSTS & MACROS>------
//------<STRUCTS>------

struct UniformBufferObject{
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
};

struct Offscreen{
	int32_t width, height;
	VkFramebuffer framebuffer;
	Texture baseColor, maskColor, depth;
	VkRenderPass renderPass;
	VkPipelineLayout pipelineLayout;
	VkPipeline graphicsPipeline;
	Buffer uniformB;
	VkDescriptorPool descriptorPool;
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSet descriptorSet;
	VkCommandBuffer commandBuffer;
} offscreen;

//------</STRUCTS>------
//------<HELPERS>------
//------ <CLASS VULKANOFFSCREEN> ------
class VulkanOffscreen : public VulkanBase {

public:

	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanupOffscreen();
		cleanup();
	}

//------<MEMBERS>------
private:
	//descriptor layout
	//to pass the transformation matrix to the vertex shader
	VkPipelineLayout pipelineLayout;
	
//------</MEMBERS>------
//------<INIT VULKAN>------
	void initVulkan() {
		initBase();
		prepareOffscreen();
	}

//------</INIT VULKAN>------
//------<GRAPHICS PIPELINE>------
	void createGraphicsPipeline(VkRenderPass& renderPass, VkPipeline& graphicsPipeline, VkPipelineLayout& pipelineLayout){

		auto vertShaderCode = readFile("shaders_offscreen/vert.spv");
		auto fragShaderCode = readFile("shaders_offscreen/frag.spv");	
		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		//to use shaders assign them to specific pipeline stage
		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		//specify module containing the code
		vertShaderStageInfo.module = vertShaderModule;
		//specify function to invoke
		vertShaderStageInfo.pName = "main";

		//same for frag shader
		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		//used to ref them in actual pipeline creation step
		VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

		//FIXED FUNCTIONS PART
		//VERTEX INPUT
		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		//modified for struct Vertex
		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

		//binding: spacing between data and whether the data is per-vertex or per-instance
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		//attr: type of the attributes passed to the vertex shader, which binding to load them from and at which offset
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription; // Optional
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data(); // Optional
	
		//INPUT ASSEMBLY
		//define topology: do vertices form points/lines/triangles
		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		//VIEWPORTS & SCISSORS
		//viewport defines transform from image to framebuffer
		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float) swapChainExtent.width;
		viewport.height = (float) swapChainExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		//scissor rectangles define in which regions pixels will be stored (a filter)
		VkRect2D scissor{};
		scissor.offset = {0, 0};
		scissor.extent = swapChainExtent;
		//combine them into a viewport state
		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;

		//RASTERIZER
		//takes geometry shaped by vertices from vert shader and turns it info fragments to be coloured by frag shader.
		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		//If fragment fills whole polygon/ just edge/point
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;

		rasterizer.lineWidth = 3.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f; // Optional
		rasterizer.depthBiasClamp = 0.0f; // Optional
		rasterizer.depthBiasSlopeFactor = 0.0f; // Optional
		
		//MULTISAMPLING
		//combine fragment shader results of multiple polygons that rasterize to the same pixel
		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f; // Optional
		multisampling.pSampleMask = nullptr; // Optional
		multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
		multisampling.alphaToOneEnable = VK_FALSE; // Optional

		//DEPTH
		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		//if depth of new frags should be compared to depth buffer to see if they should be discarded
		depthStencil.depthTestEnable = VK_TRUE;
		//if new depths from frags that pass test should be written to depth buffer
		depthStencil.depthWriteEnable = VK_TRUE;
		//comparison operation. lower depth <==> closer
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		//used for optional depth bound test (only use frags whose depth falls in specified range)
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.minDepthBounds = 0.0f; // Optional
		depthStencil.maxDepthBounds = 1.0f; // Optional
		depthStencil.stencilTestEnable = VK_FALSE;

		//COLOR BLENDING
		//combine color from frag shader w color already in framebuffer
		std::array<VkPipelineColorBlendAttachmentState, 2> colorBlendAttachments {};
		colorBlendAttachments[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachments[0].blendEnable = VK_FALSE;
		colorBlendAttachments[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachments[1].blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
		colorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size());
		colorBlending.pAttachments = colorBlendAttachments.data();
		colorBlending.blendConstants[0] = 0.0f; // Optional
		colorBlending.blendConstants[1] = 0.0f; // Optional
		colorBlending.blendConstants[2] = 0.0f; // Optional
		colorBlending.blendConstants[3] = 0.0f; // Optional

		//DYNAMIC STATE (MISSING)


		//PIPELINE LAYOUT
		//to pass the transformation matrix to the vertex shader
		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1; // Optional
		pipelineLayoutInfo.pSetLayouts = &offscreen.descriptorSetLayout; // Optional
		//push constants for mask color
		VkPushConstantRange pushConstantRange{};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = static_cast<uint32_t>(sizeof(glm::vec3));
		pipelineLayoutInfo.pushConstantRangeCount = 1; // Optional
		pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange; // Optional

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS){
			throw std::runtime_error("failed to create pipeline layout!");
		}

		//create graphics pipeline
		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		//reference fixed-function stage
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil; // Optional
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = nullptr; // Optional
		//ref pipeline layout
		pipelineInfo.layout = pipelineLayout;
		//ref render pass & subpass index
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;

		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
		pipelineInfo.basePipelineIndex = -1; // Optional

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS){
			throw std::runtime_error("failed to create graphics pipeline!");
		}


		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
	}
//------</GRAPHICS PIPELINE>------
//---------------------------------OFFSCREEN------------------------------------------------------

	void updateUniformBufferOffscreen(){
			
		globalTime+=0.03;
		auto time = globalTime;
		
		UniformBufferObject ubo{};
		ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(45.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view = glm::lookAt(glm::vec3(3.0f, 1.0f, 0.7f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.proj = glm::perspective(glm::radians(30.0f), offscreen.width / (float) offscreen.height, 0.1f, 10.0f);
		ubo.proj[1][1] *= -1;

		void* data;
		vkMapMemory(device, offscreen.uniformB.mem, 0, sizeof(ubo), 0, &data);
		memcpy(data, &ubo, sizeof(ubo));
		vkUnmapMemory(device, offscreen.uniformB.mem);
	}


	void createDescriptorPoolOffscreen(){
		std::array<VkDescriptorPoolSize, 2> poolSizes{};
		//uniform buffer
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = 1;
		//image sampler
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[1].descriptorCount = 1;

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = 1;
		
		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &offscreen.descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create offscreen descriptor pool!");
		}
	}


	void createDescriptorSetLayoutOffscreen(){

		//universal buffer descriptor
		VkDescriptorSetLayoutBinding uboLayoutBinding{};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.pImmutableSamplers = nullptr; // Optional
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		//combined image sampler descriptor
		VkDescriptorSetLayoutBinding samplerLayoutBinding{};
		samplerLayoutBinding.binding = 1;
		samplerLayoutBinding.descriptorCount = 1;
		samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		samplerLayoutBinding.pImmutableSamplers = nullptr;
		samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};
		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &offscreen.descriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}
	}


	void createDescriptorSetOffscreen(){
		//how many sets to allocate, based on what descriptor layout	
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = offscreen.descriptorPool;
		//one descriptor set for each swap image, all with same layout
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &offscreen.descriptorSetLayout;

		//allocate sets
		if (vkAllocateDescriptorSets(device, &allocInfo, &offscreen.descriptorSet) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate dump descriptor sets!");
		}

		//populate
		//uniform buffer
		VkDescriptorBufferInfo bufferInfo{};
		bufferInfo.buffer = offscreen.uniformB.buffer;
		bufferInfo.offset = 0;
		bufferInfo.range = sizeof(UniformBufferObject);

		//image sampler
		VkDescriptorImageInfo imageInfo{};
		imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageInfo.imageView = motif.view;
		imageInfo.sampler = motif.sampler;

		//write descriptors
		std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

		//uniform buffer
		descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		//destination
		descriptorWrites[0].dstSet = offscreen.descriptorSet;
		descriptorWrites[0].dstBinding = 0;
		descriptorWrites[0].dstArrayElement = 0;
		descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorWrites[0].descriptorCount = 1;
		descriptorWrites[0].pBufferInfo = &bufferInfo;
		descriptorWrites[0].pImageInfo = nullptr; // Optional
		descriptorWrites[0].pTexelBufferView = nullptr; // Optional

		//image sampler
		descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		//destination
		descriptorWrites[1].dstSet = offscreen.descriptorSet;
		descriptorWrites[1].dstBinding = 1;
		descriptorWrites[1].dstArrayElement = 0;
		descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		descriptorWrites[1].descriptorCount = 1;
		descriptorWrites[1].pImageInfo = &imageInfo;

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
	}


	//specify framebuffer attachments that will be used while rendering. Pe how many color and depth buffers there will be
	void createRenderPassOffscreen() {
		std::array<VkAttachmentDescription, 3> attachmentDescs = {};
		for(uint32_t i=0; i<3; i++){
			attachmentDescs[i].samples = VK_SAMPLE_COUNT_1_BIT;
			attachmentDescs[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
			attachmentDescs[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			attachmentDescs[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			attachmentDescs[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;	
			attachmentDescs[i].initialLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			attachmentDescs[i].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			if(i==2){ //depth
				attachmentDescs[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
				attachmentDescs[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
				attachmentDescs[i].initialLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
				attachmentDescs[i].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			}
		}

		//format
		attachmentDescs[0].format = COLOR_FORMAT;
		attachmentDescs[1].format = COLOR_FORMAT;
		attachmentDescs[2].format = findDepthFormat();


		//add ref to color attachment for subpasses
		std::vector<VkAttachmentReference> colorReferences;
		colorReferences.push_back({0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
		colorReferences.push_back({1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

		//add ref to depth attachment.
		VkAttachmentReference depthReference{};
		depthReference.attachment = 2;
		depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		//describe subpass
		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = static_cast<uint32_t>(colorReferences.size());
		subpass.pColorAttachments = colorReferences.data();
		subpass.pDepthStencilAttachment = &depthReference;

		//adding dependency as per 'rend and pres'
		std::array<VkSubpassDependency, 2> dependencies;

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		//wait for color attachment
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		//wait for color attachment
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;


		//create render pass
		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachmentDescs.size());
		renderPassInfo.pAttachments = attachmentDescs.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = 2;
		renderPassInfo.pDependencies = dependencies.data();

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &offscreen.renderPass) != VK_SUCCESS){
			throw std::runtime_error("failed to create offscreen render pass!");
		}

	}


	void createFramebufferOffscreen() {

			std::array<VkImageView, 3> attachments = {
				 offscreen.baseColor.view,
				 offscreen.maskColor.view,
				 offscreen.depth.view
			};

			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = offscreen.renderPass;
			framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			framebufferInfo.pAttachments = attachments.data();
			framebufferInfo.width = offscreen.width;
			framebufferInfo.height = offscreen.height;
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &offscreen.framebuffer) != VK_SUCCESS) {
				 throw std::runtime_error("failed to create offscreen framebuffer!");
			}
	}


	void renderObject(uint32_t objectIndex, VkCommandBuffer& commandBuffer){
			
			//reuse offscreen render pass
			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = offscreen.renderPass;
			renderPassInfo.framebuffer = offscreen.framebuffer;
			renderPassInfo.renderArea.offset = {0, 0};
			renderPassInfo.renderArea.extent.width = offscreen.width;
			renderPassInfo.renderArea.extent.height= offscreen.height;

			std::array<VkClearValue, 3> clearValues{};
			clearValues[0].color = {0.0f, 0.0f, 0.0f, 1.0f};
			clearValues[1].color = {0.0f, 0.0f, 0.0f, 1.0f};
			clearValues[2].depthStencil = {1.0f, 0};
			renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
			renderPassInfo.pClearValues = clearValues.data();

			//color of mask of object. Object starts at indexOffsetNum and is defined by indicesNum indices.
			glm::vec3 maskColor;
			uint32_t indexOffsetNum, indicesNum;
			switch (objectIndex){
				case 0: //floor
					maskColor = glm::vec3(0.0f, 1.0f, 0.0f);
					indexOffsetNum = 0;
					indicesNum = 12;
					break;
				case 1: //cube
					maskColor = glm::vec3(0.0f, 0.0f, 0.1f);
					indexOffsetNum = 12;
					indicesNum = 72;
					break;
				case 2: //pyramid
					maskColor = glm::vec3(0.1f, 0.0f, 0.0f);
					indexOffsetNum = 84;
					indicesNum = 36;
					break;
			}


			vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
			vkCmdPushConstants(commandBuffer, offscreen.pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::vec3), &maskColor);


			//clear attachments at beginning of every run
			std::array<VkClearAttachment, 3> clearAttachments{};
			clearAttachments[0].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			clearAttachments[0].colorAttachment = 0;
			clearAttachments[0].clearValue = clearValues[0];
			clearAttachments[1].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			clearAttachments[1].colorAttachment = 1;
			clearAttachments[1].clearValue = clearValues[1];
			clearAttachments[2].aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
			clearAttachments[2].clearValue = clearValues[2];

			VkRect2D rectToClear{};
			rectToClear.offset = {0, 0};
			rectToClear.extent.width = offscreen.width;
			rectToClear.extent.height = offscreen.height;

			std::array<VkClearRect, 3> clearRects{};
			for(int i=0; i<3; i++){
				clearRects[i].rect = rectToClear;
				clearRects[i].baseArrayLayer = 0;
				clearRects[i].layerCount = 1;
			}

			if(indexOffsetNum == 0){
				vkCmdClearAttachments(commandBuffer, static_cast<uint32_t>(clearAttachments.size()), clearAttachments.data(), static_cast<uint32_t>(clearRects.size()), clearRects.data());
			}

			//bind pipeline
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, offscreen.graphicsPipeline);

			VkBuffer vertexBuffers[] = {vertexB.buffer};
			VkDeviceSize offsets[] = {0};
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

			VkDeviceSize indexOffset = sizeof(indices[0]) * indexOffsetNum;
			vkCmdBindIndexBuffer(commandBuffer, indexB.buffer, indexOffset, VK_INDEX_TYPE_UINT16);

			//bind descriptor set for each swap chain image to the descriptors in shader
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, offscreen.pipelineLayout, 0, 1, &offscreen.descriptorSet, 0, nullptr);

			vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indicesNum), 1, 0, 0, 0);

			vkCmdEndRenderPass(commandBuffer);

	}


	void createCommandBufferOffscreen() {

		//initialise
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = 1;

		if (vkAllocateCommandBuffers(device, &allocInfo, &offscreen.commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate offscreen command buffer!");
		}

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		if (vkBeginCommandBuffer(offscreen.commandBuffer, &beginInfo) != VK_SUCCESS) {
			 throw std::runtime_error("failed to begin recording offscreen command buffer!");
		}

		//do actual stuff
		for(int i=0; i<3; i++){ //for every object
			renderObject(i, offscreen.commandBuffer);
		}
		

		if (vkEndCommandBuffer(offscreen.commandBuffer) != VK_SUCCESS) {
			 throw std::runtime_error("failed to record offscreen command buffer!");
		}

	}


	void prepareOffscreen(){
		offscreen.width = WIDTH;
		offscreen.height = HEIGHT;
			
		//create base color image & view
		createImage(offscreen.width, offscreen.height, COLOR_FORMAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, offscreen.baseColor.image, offscreen.baseColor.mem);
		offscreen.baseColor.view = createImageView(offscreen.baseColor.image, COLOR_FORMAT, VK_IMAGE_ASPECT_COLOR_BIT);
		transitionImageLayout(offscreen.baseColor.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, 1);
		//create mask color image & view
		createImage(offscreen.width, offscreen.height, COLOR_FORMAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, offscreen.maskColor.image, offscreen.maskColor.mem);
		offscreen.maskColor.view = createImageView(offscreen.maskColor.image, COLOR_FORMAT, VK_IMAGE_ASPECT_COLOR_BIT);
		transitionImageLayout(offscreen.maskColor.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, 1);
		//create depth image & view
		VkFormat depthFormat = findDepthFormat();
		createImage(offscreen.width, offscreen.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, offscreen.depth.image, offscreen.depth.mem);
		offscreen.depth.view = createImageView(offscreen.depth.image, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
		transitionImageLayout(offscreen.depth.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT, 1); 


















		//create uniform buffer
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);
		createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, offscreen.uniformB.buffer, offscreen.uniformB.mem);

		//create descriptor pool
		createDescriptorPoolOffscreen();
		//create descriptor set layout
		createDescriptorSetLayoutOffscreen();
		//create descriptor set
		createDescriptorSetOffscreen();

		//create render pass
		createRenderPassOffscreen();

		//create graphics pipeline
		createGraphicsPipeline(offscreen.renderPass, offscreen.graphicsPipeline, offscreen.pipelineLayout);

		//create framebuffer
		createFramebufferOffscreen();

		//create command buffer
		createCommandBufferOffscreen();
	}

	void dumpOffscreen() {
		std::string basePathPrefix = DUMP_PATH+"offscreen_base";
		std::string maskPathPrefix = DUMP_PATH+"offscreen_mask";
		std::string ext = ".ppm";

		std::cout << "dumping.." << std::endl << std::endl;


		int runs=3;
		for(int n=0; n<runs; n++){
			updateUniformBufferOffscreen();

			VkSubmitInfo submitInfo{};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &offscreen.commandBuffer;

			if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
				throw std::runtime_error("failed to submit offscreen command buffer!");
			}

			std::string basePath = basePathPrefix;
			std::string maskPath = maskPathPrefix;
			int lenDif = std::to_string(runs).size() - std::to_string(n).size();
			while(lenDif){
				basePath += "0";
				maskPath += "0";
				lenDif--;
			}
			basePath += std::to_string(n) + ext;
			maskPath += std::to_string(n) + ext;

			saveOutputColorTexture(basePath, offscreen.baseColor.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, COLOR_FORMAT, offscreen.width, offscreen.height);
			saveOutputColorTexture(maskPath, offscreen.maskColor.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, COLOR_FORMAT, offscreen.width, offscreen.height);
		}

		std::cout << "Screenshots dumped to disk at: " << DUMP_PATH << std::endl << std::endl;

	}

	void presentOffscreen(){
		
		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, nullptr, VK_NULL_HANDLE, &imageIndex);

		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			std::cout << "OUT OF DATE KHR" << std::endl << std::endl;
			return;
		} else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		blit(offscreen.baseColor.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, COLOR_FORMAT, WIDTH, HEIGHT, swapChainImages[imageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_ASPECT_COLOR_BIT, 1);

		//configure presentation
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		VkSwapchainKHR swapChains[] = {swapChain};
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;

		//to check result in case of multiple swap chains
		presentInfo.pResults = nullptr; // Optional

		result = vkQueuePresentKHR(presentQueue, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
			framebufferResized = false;
			std::cout << "OUT OF DATE KHR" << std::endl << std::endl;
		} else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to present swap chain image!");
		}

		//crude; forces only 1 frame in pipeline
		vkQueueWaitIdle(presentQueue);

	}

	void cleanupOffscreen(){
		//framebuffers
		vkDestroyFramebuffer(device, offscreen.framebuffer, nullptr);

		//command buffer
		vkFreeCommandBuffers(device, commandPool, 1, &offscreen.commandBuffer);

		//baseColor
		vkDestroyImageView(device, offscreen.baseColor.view, nullptr);
		vkFreeMemory(device, offscreen.baseColor.mem, nullptr);
		vkDestroyImage(device, offscreen.baseColor.image, nullptr);
		//maskColor
		vkDestroyImageView(device, offscreen.maskColor.view, nullptr);
		vkFreeMemory(device, offscreen.maskColor.mem, nullptr);
		vkDestroyImage(device, offscreen.maskColor.image, nullptr);
		//depth
		vkDestroyImageView(device, offscreen.depth.view, nullptr);
		vkFreeMemory(device, offscreen.depth.mem, nullptr);
		vkDestroyImage(device, offscreen.depth.image, nullptr);

		//uniform buffer
		vkDestroyBuffer(device, offscreen.uniformB.buffer, nullptr);
		vkFreeMemory(device, offscreen.uniformB.mem, nullptr);

		//descriptor pool
		vkDestroyDescriptorPool(device, offscreen.descriptorPool, nullptr);

		//render pass
		vkDestroyRenderPass(device, offscreen.renderPass, nullptr);

		//graphics pipelines
		vkDestroyPipeline(device, offscreen.graphicsPipeline, nullptr);
		//pipeline layout
		vkDestroyPipelineLayout(device, offscreen.pipelineLayout, nullptr);

	}
//------</DUMP>------
//------<>------
//------<>------
//------<MAIN LOOP>------
	void mainLoop() {
		dumpOffscreen();


/*
		while(!glfwWindowShouldClose(window)){
			glfwPollEvents();
			presentOffscreen();
		}

*/

		vkDeviceWaitIdle(device);

	}

//------</MAIN LOOP>------
//------</CLEANUP>------

	void cleanup() {

		//swap chain
		vkDestroySwapchainKHR(device, swapChain, nullptr);

		//descriptor layout
		vkDestroyDescriptorSetLayout(device, offscreen.descriptorSetLayout, nullptr);
			


	}
//------</CLEANUP>------
};
//------ </CLASS VULKANOFFSCREEN> ------


//------<MAIN>------
int main() {
	VulkanOffscreen app;

	try {
		app.run();
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
//------</MAIN>------

