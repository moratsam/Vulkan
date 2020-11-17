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
#include <unistd.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "vulkanbase.h"

//------<CONSTS & MACROS>------
const std::string DUMP_PATH = "./dump/";
int screenshotNum = 0;

const int MAX_FRAMES_IN_FLIGHT = 2;

//------</CONSTS & MACROS>------
//------<STRUCTS>------
struct UniformBufferObject{
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
};

//------</STRUCTS>------
//------ <CLASS SWAPCHAIN> ------
class SwapChain : public VulkanBase {

public:

	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

//------<MEMBERS>------
private:
	
//to store image views
	std::vector<VkImageView> swapChainImageViews;
	//to hold framebuffers
	std::vector<VkFramebuffer> swapChainFramebuffers;

	//handle for render pass
	VkRenderPass renderPass;
	//descriptor layout
	VkDescriptorSetLayout descriptorSetLayout;
	//to pass the transformation matrix to the vertex shader
	VkPipelineLayout pipelineLayout;
	//pipeline handle
	VkPipeline graphicsPipeline;


	//uniform Buffer per swap chain image
	std::vector<Buffer> uniformB;
	
	//descriptor pool
	VkDescriptorPool descriptorPool;
	std::vector<VkDescriptorSet> descriptorSets;

	//command buffer for every image in swapchain
	std::vector<VkCommandBuffer> commandBuffers;
	
	//semaphore to signal ready for rendering, presentation
	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	//fence for each frame
	std::vector<VkFence> inFlightFences;
	//fence for each frame in flight
	std::vector<VkFence> imagesInFlight;
	//index to keep track of current frame
	size_t currentFrame = 0;
	

//------</MEMBERS>------
//------<INIT VULKAN>------
	void initVulkan() {
		initBase();
		createSwapChainImageViews();
		createRenderPass();
		createDescriptorSetLayout();
		createGraphicsPipeline();
		createDepthResources();
		createSwapChainFramebuffers();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffers();
		createSyncObjects();

/*
*/
	}

//------</INIT VULKAN>------
//------<SWAPCHAIN>------
	void recreateSwapChain() {
		//in case of minimization (win size 0), pause (wait for window to get back to foreground).
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(device);

		cleanupSwapChain();

		createSwapChain();
		createSwapChainImageViews();
		createRenderPass();
		createGraphicsPipeline();
		createDepthResources();
		createSwapChainFramebuffers();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffers();
	}

//------</SWAPCHAIN>------
//------<IMAGE VIEWS>------
	void createSwapChainImageViews() {
		swapChainImageViews.resize(swapChainImages.size());

		for (uint32_t i = 0; i < swapChainImages.size(); i++) {
			swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
		}
	}

	
//------</IMAGE VIEWS>------
//------<RENDER PASS>------
	//specify framebuffer attachments that will be used while rendering. Pe how many color and depth buffers there will be
	void createRenderPass() {
		//color attachment
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = swapChainImageFormat;
		//count 1 becaue not multisampling
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		//what to do with color,depth data before,after rendering
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		//what to do with stencil data
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		//which layout image will have before render pass
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		//transition to layout after render pass - ready for PRESENTATION
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		//depth attachment
		VkAttachmentDescription depthAttachment{};
		//format should be same as depth image itself
		depthAttachment.format = findDepthFormat();
		depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		//dont care about storing depth data, will not be used after drawing has finished.
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		//dont care about previous depth contents
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		//add ref to color attachment for subpasses
		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		//add ref to attachment for first (and only) subpass.
		VkAttachmentReference depthAttachmentRef{};
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		//describe subpass
		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;
		subpass.pDepthStencilAttachment = &depthAttachmentRef;

		//adding dependency as per 'rend and pres'
		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		//wait for color attachment
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		//create render pass
		std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};
		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS){
			throw std::runtime_error("failed to create render pass!");
		}

	}


//------</RENDER PASS>------
//------<GRAPHICS PIPELINE>------
	void createGraphicsPipeline(){
		auto vertShaderCode = readFile("shaders_swapchain/vert.spv");
		auto fragShaderCode = readFile("shaders_swapchain/frag.spv");
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
		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
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
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout; // Optional
		pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
		pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

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
//------<FRAMEBUFFERS>------
	void createSwapChainFramebuffers() {
		swapChainFramebuffers.resize(swapChainImageViews.size());

		for (size_t i = 0; i < swapChainImageViews.size(); i++) {
			std::array<VkImageView, 2> attachments = {
				swapChainImageViews[i],
				depth.view
			};

			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = renderPass;
			//specify the VkImageView-s that should be bound to the respective attachment descriptions in the render pass pAttachment array
			framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			framebufferInfo.pAttachments = attachments.data();
			framebufferInfo.width = swapChainExtent.width;
			framebufferInfo.height = swapChainExtent.height;
			//number of layers in image arrays
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS){
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}

//------</FRAMEBUFFERS>------
//------<COMMAND BUFFER>------
	void createCommandBuffers() {
		commandBuffers.resize(swapChainFramebuffers.size());
	
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();

		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS){
			throw std::runtime_error("failed to allocate command buffers!");
		}

		//record command buffers
		for (size_t i = 0; i < commandBuffers.size(); i++) {
			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = 0; // Optional
			beginInfo.pInheritanceInfo = nullptr; // Optional

			if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS){
				throw std::runtime_error("failed to begin recording command buffer!");
			}
	
			//drawing starts by beginning the render pass
			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = renderPass;
			renderPassInfo.framebuffer = swapChainFramebuffers[i];
			renderPassInfo.renderArea.offset = {0, 0};
			renderPassInfo.renderArea.extent = swapChainExtent;

			std::array<VkClearValue, 2> clearValues{};
			clearValues[0].color= {0.0f, 0.0f, 0.0f, 1.0f};
			clearValues[1].depthStencil = {1.0f, 0};

			renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
			renderPassInfo.pClearValues = clearValues.data();

			vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
			//bind graphics pipeline
			vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

			VkBuffer vertexBuffers[] = {vertexB.buffer};
			VkDeviceSize offsets[] = {0};
			vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);
			vkCmdBindIndexBuffer(commandBuffers[i], indexB.buffer, 0, VK_INDEX_TYPE_UINT16);

			//bind descriptor set for each swap chain image to the descriptors in shader
			vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);
			//draw
			//vkCmdDraw(commandBuffers[i], static_cast<uint32_t>(vertices.size()), 1, 0, 0);
			vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
			
			//end render pass
			vkCmdEndRenderPass(commandBuffers[i]);
		
			//finish recording
			if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS){
				throw std::runtime_error("failed to record command buffer!");
			}
		}
	}
//------</COMMAND BUFFER>------
//------<SYNC OBJECTS>------
	void createSyncObjects(){
		//resize vectors
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
		imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for(size_t i=0; i<MAX_FRAMES_IN_FLIGHT; i++){
			
			if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS || \
					vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS || \
					vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS){
	 			throw std::runtime_error("failed to create synchronization objects for a frame!");
	 		}
		}
	}
//------</SYNC OBJECTS>------
//------<DRAW FRAME>------
	void drawFrame(){
		//wait for fence from QueueSubmit
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
		
		//to store image index
		uint32_t imageIndex;
		//result to see if swap chain needs to be recreated because of window resize
		VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			recreateSwapChain();
			return;
		} else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		//check if previous frame is using this image
		if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
			vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
		}
		// Mark the image as now being in use by this frame
		imagesInFlight[imageIndex] = inFlightFences[currentFrame];

		updateUniformBuffer(imageIndex);

		//which semaphores to wait on before execution begins and in which stage(s) of the pipeline to wait.
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
		VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;

		//submit buffers for execution
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

		//which semaphores to signal once command buffer finished execution
		VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		//reset fence
		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		//submit command buffer to graphics queue
		//fence par to signal command buffer has finished executing
		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS){
			throw std::runtime_error("failed to submit draw command buffer!");
		}


		//configure presentation
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapChains[] = {swapChain};
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;

		//to check result in case of multiple swap chains
		presentInfo.pResults = nullptr; // Optional

		result = vkQueuePresentKHR(presentQueue, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
			framebufferResized = false;
			recreateSwapChain();
		} else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to present swap chain image!");
		}

		//crude; forces only 1 frame in pipeline
		//vkQueueWaitIdle(presentQueue);

		currentFrame = (1+currentFrame)%MAX_FRAMES_IN_FLIGHT;
	}

//------</DRAW FRAME>------
//------<BUFFERS>------
	void createUniformBuffers(){
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);

		uniformB.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformB[i].buffer, uniformB[i].mem);
		}
	}

	void updateUniformBuffer(uint32_t currentImage){
		
		static auto startTime = std::chrono::high_resolution_clock::now();

		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
		
		UniformBufferObject ubo{};
		ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(45.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view = glm::lookAt(glm::vec3(3.0f, 3.0f, 3.5f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.proj = glm::perspective(glm::radians(30.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 10.0f);
		ubo.proj[1][1] *= -1;

		void* data;
		vkMapMemory(device, uniformB[currentImage].mem, 0, sizeof(ubo), 0, &data);
		memcpy(data, &ubo, sizeof(ubo));
		vkUnmapMemory(device, uniformB[currentImage].mem);

		if (glfwGetKey(window, GLFW_KEY_F3) == GLFW_PRESS) {
			std::string dump_fname = DUMP_PATH + "screenshot" + std::to_string(screenshotNum) + ".ppm";
			saveOutputColorTexture(dump_fname, swapChainImages[currentImage], VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, swapChainImageFormat, swapChainExtent.width, swapChainExtent.height);
			screenshotNum++;
			std::cout << "Screenshots dumped to disk at: " << DUMP_PATH << std::endl << std::endl;
			usleep(100*1000); //microseconds
		}

	}

//------</BUFFERS>------
//------<DESCRIPTOR>------
	void createDescriptorSetLayout(){

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

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}
	}

	void createDescriptorPool(){
		std::array<VkDescriptorPoolSize, 2> poolSizes{};
		//uniform buffer
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		//image sampler
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());
		
		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}


	void createDescriptorSets(){
		//how many sets to allocate, based on what descriptor layout	
		std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		//one descriptor set for each swap image, all with same layout
		allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
		allocInfo.pSetLayouts = layouts.data();

		//allocate sets
		descriptorSets.resize(swapChainImages.size());
		if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		//populate
		for (size_t i = 0; i < swapChainImages.size(); i++) {
			//uniform buffer
			VkDescriptorBufferInfo bufferInfo{};
			bufferInfo.buffer = uniformB[i].buffer;
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
			descriptorWrites[0].dstSet = descriptorSets[i];
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
			descriptorWrites[1].dstSet = descriptorSets[i];
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pImageInfo = &imageInfo;

			vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
		}
	}

//------</DESCRIPTOR>------
//------<>------
//------<>------
//------<MAIN LOOP>------
	void mainLoop() {
	/*
	*/
		while(!glfwWindowShouldClose(window)){
			glfwPollEvents();
			drawFrame();
		}
		//dump();
		vkDeviceWaitIdle(device);

	}

//------</MAIN LOOP>------
//------</CLEANUP>------
	void cleanupSwapChain(){

		//depth Texture
		vkDestroyImageView(device, depth.view, nullptr);
		vkDestroyImage(device, depth.image, nullptr);
		vkFreeMemory(device, depth.mem, nullptr);
		//framebuffers 
		for (auto framebuffer : swapChainFramebuffers){
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}
		//could recreate command pool, but it's leaner to just reuse existing pool to allocate new command buffers
		vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
		//graphics pipeline
		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		//pipeline layout
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		//render pass
		vkDestroyRenderPass(device, renderPass, nullptr);
		//image views
		for (auto imageView : swapChainImageViews){
			vkDestroyImageView(device, imageView, nullptr);
		}
		//swap chain
		vkDestroySwapchainKHR(device, swapChain, nullptr);
		//uniform buffers
		for (size_t i = 0; i < swapChainImages.size(); i++) {
			vkDestroyBuffer(device, uniformB[i].buffer, nullptr);
			vkFreeMemory(device, uniformB[i].mem, nullptr);
		}
		//descriptor pool
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);

	}

	void cleanup() {
		cleanupSwapChain();

		//descriptor layout
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
		//sync objects
		for(size_t i=0; i<MAX_FRAMES_IN_FLIGHT; i++){
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}

		cleanupBase();
	}

//------</CLEANUP>------
};
//------ </CLASS SWAPCHAIN> ------


//------<MAIN>------
int main() {
	SwapChain app;

	try {
		app.run();
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
//------</MAIN>------
