#include "vertices.h"

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

#ifdef NDEBUG
	const bool enableValidationLayers = false;
#else
	const bool enableValidationLayers = true;
#endif

extern const std::vector<uint16_t> indices;
extern const std::vector<Vertex> vertices;

//------<STRUCTS>------

struct Buffer{
	VkBuffer buffer;
	VkDeviceMemory mem;
};

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

struct QueueFamilyIndices {
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;
	 
	 bool isComplete() {
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

struct Texture{
	VkImage image;
	VkDeviceMemory mem;
	VkImageView view;
	VkSampler sampler;
};

const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

//------</STRUCTS>------
//------<HELPERS>------
bool checkValidationLayerSupport(){
	uint32_t layerCount;
	vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

	std::vector<VkLayerProperties> availableLayers(layerCount);
	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

	for(const char* layerName : validationLayers){
		bool layerFound = false;
		for(const auto& layerProperties : availableLayers){
			if (strcmp(layerName, layerProperties.layerName) == 0){
				layerFound = true;
				break;
			}
		}
		if(!layerFound){
			return false;
		}
	}
	return true;
}

//VALIDATION LAYER
//return the required list of extensions based on whether validation
//layers are enabled or not.
std::vector<const char*> getRequiredExtensions() {
	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions;
	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

	if (enableValidationLayers) {
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

	return extensions;
}

//DEBUG
//debug callback function
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData) {

	std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl << std::endl;

	return VK_FALSE;
}

//DEBUG
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	} else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

//DEBUG
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}

//LOAD SHADERS
static std::vector<char> readFile(const std::string& filename) {
	//start reading in binary at end of file
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()){
		throw std::runtime_error("failed to open file!");
	}
	//use read position to define necessary buffer size
	size_t fileSize = (size_t) file.tellg();
	std::vector<char> buffer(fileSize);
	//seek to beginning of file and read all bytes at once
	file.seekg(0);
	file.read(buffer.data(), fileSize);
	//close file
	file.close();

	return buffer;
}

//------</HELPERS>------











//--------------------<CLASS>------
class VulkanBase{
	//MEMBERS
	public:
	GLFWwindow* window;
	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkSurfaceKHR surface;

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	//logdev
	VkDevice device;
	
	//handle to interface with queue automatically created w logdev
	VkQueue graphicsQueue;
	//presentation handle
	VkQueue presentQueue;
	//swapchain
	VkSwapchainKHR swapChain;
	//images handle
	std::vector<VkImage> swapChainImages;
	//format and extent of swapchain images
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	
	//manage memory for buffers
	VkCommandPool commandPool;
		
	//depth
	Texture depth;
	//motif image
	Texture motif;

	//vertex Buffer
	Buffer vertexB;
	//index Buffer
	Buffer indexB;


	//flag that window resize happened
	bool framebufferResized = false;


//------<INIT VULKAN>------
	void initBase(){
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createCommandPool();
		createMotifResources();
		createVertexBuffer();
		createIndexBuffer();
	}

//------</INIT VULKAN>------
//------<INSTANCE>------
	void createInstance(){
		//validation
		if(enableValidationLayers && !checkValidationLayerSupport()){
			throw std::runtime_error("validation layers requested, but not available!");
		}

		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		auto extensions = getRequiredExtensions();
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
		if(enableValidationLayers){
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else{
			createInfo.enabledLayerCount = 0; 
			createInfo.pNext = nullptr;
		}
			
		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
			throw std::runtime_error("failed to create instance!");
		}
	}

//------</INSTANCE>------
//------<INIT WINDOW>------
	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<VulkanBase*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}


	void initWindow(){
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		//glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

	}

//------</INIT WINDOW>------
//------<DEBUG>------
void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
	createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |\
											VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |\
											VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |\
										VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |\
										VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	createInfo.pfnUserCallback = debugCallback;
}


	void setupDebugMessenger(){
		if (!enableValidationLayers) return;
		
		VkDebugUtilsMessengerCreateInfoEXT createInfo{};
		populateDebugMessengerCreateInfo(createInfo);
		createInfo.pUserData = nullptr; // Optional

		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger!");
		}
	}

//------</DEBUG>------
//------<SURFACE>------
	void createSurface(){
		if(glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS){
			throw std::runtime_error("failed to create window surface!");
		}
	}

//------</SURFACE>------
//------<PHYSICAL DEVICE>------
	//We need to check which queue families are supported by the device and which one of these supports the commands that we want to use.
	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device){
		QueueFamilyIndices indices;
		
		// Assign index to queue families that could be found
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT){
				indices.graphicsFamily = i;
			}
			//check Qfam has capability of presenting to window surface
			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
			if(presentSupport){
				indices.presentFamily = i;
			}
			if(indices.isComplete()){
				break;
			}
			i++;
		}
		return indices;
	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice device){
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();

	}

	bool isDeviceSuitable(VkPhysicalDevice device){
		//basic query of name, type and supported Vulkan version etc
		VkPhysicalDeviceProperties deviceProperties;
		//query optional features like texture compression, 64 bit floats
		VkPhysicalDeviceFeatures deviceFeatures;

		vkGetPhysicalDeviceProperties(device, &deviceProperties);
		vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

		//Because we're just starting out, Vulkan support is the only thing we need
		//and therefore we'll settle for just any GPU:
		QueueFamilyIndices indices = findQueueFamilies(device);
		
		//add check if extensions (swapchain) are supported
		bool extensionsSupported = checkDeviceExtensionSupport(device);

		//add check if swapchain support is adequate
		bool swapChainAdequate = false;
		if (extensionsSupported) {
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}


		return indices.isComplete() && extensionsSupported && swapChainAdequate && deviceFeatures.samplerAnisotropy;
	}
	

	void pickPhysicalDevice(){
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
		if(deviceCount == 0){
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}
		//retrieve graphical cards and pick first suitable	
		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		for(const auto& device : devices){
			if(isDeviceSuitable(device)){
				physicalDevice = device;
				break;
			}
		}

		if(physicalDevice == VK_NULL_HANDLE){
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

//------</PHYSICAL DEVICE>------
//------<LOGICAL DEVICE>------
	void createLogicalDevice(){
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};
		
		float queuePriority = 1.0f;
		for(uint32_t queueFamily : uniqueQueueFamilies){
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
			queueCreateInfo.queueCount = 1;
			//assign priorities to queues to influence the scheduling of command buffer execution using floating point numbers between 0.0 and 1.0.
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}
		
		//pe geometry shaders
		VkPhysicalDeviceFeatures deviceFeatures{};
		deviceFeatures.samplerAnisotropy = VK_TRUE;
		deviceFeatures.fillModeNonSolid = VK_TRUE;
		deviceFeatures.wideLines = VK_TRUE;

		//start filling the main structure
		VkDeviceCreateInfo createInfo{};
		//modify to point to vector
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		createInfo.pEnabledFeatures = &deviceFeatures;
		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();

		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		} else {
			createInfo.enabledLayerCount = 0;
		}

		//instantiate logical device
		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS){
			throw std::runtime_error("failed to create logical device!");
		}

		//get graphics queue
		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);

		//get presentation queue
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}


//------</LOGICAL DEVICE>------
//------<SWAP CHAIN>------
	//swap helper function - surface format
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats){
		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return availableFormat;
			}
		}
		//just settle for first adequate format
		return availableFormats[0];
	}


	//swap helper function - presentation mode
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
		for (const auto& availablePresentMode : availablePresentModes){
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR){
				return availablePresentMode;
			}
		}
		return VK_PRESENT_MODE_FIFO_KHR;
	}


	//swap helper function - swap extent
	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		if (capabilities.currentExtent.width != UINT32_MAX){
			return capabilities.currentExtent;
		}
		else{
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			VkExtent2D actualExtent = {
			 static_cast<uint32_t>(width),
			static_cast<uint32_t>(height)
			};

			actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
			actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

			return actualExtent;
		}
	}


	//get support details
	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
		SwapChainSupportDetails details;
		//basic surface capabilities
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		//supported surface formats
		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		//supported presentation modes
		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}


	void createSwapChain() {
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

		//set image count to 1+min
		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount){
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;
		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};
		//specify image ownership in case of several queue families
		if (indices.graphicsFamily != indices.presentFamily) {
				createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
				createInfo.queueFamilyIndexCount = 2;
				createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			createInfo.queueFamilyIndexCount = 0; // Optional
			createInfo.pQueueFamilyIndices = nullptr; // Optional
		}

		//possible transform to rotate image. Current transform is no transform.
		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;

		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;

		createInfo.oldSwapchain = VK_NULL_HANDLE;
	//create swapchain
	if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS){
		throw std::runtime_error("failed to create swap chain!");
	}

	//count images created
	vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
	//resize image handle
	swapChainImages.resize(imageCount);
	//fill image handle
	vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
	
	//store format and extent of swap chain images
	swapChainImageFormat = surfaceFormat.format;
	swapChainExtent = extent;
	}

//------</SWAP CHAIN>------
//------<IMAGE VIEWS>------
	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) {
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		//type as 1D or 3D textures and cube maps
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = format;
			//what the image purpose is and which part of image to access
		viewInfo.subresourceRange.aspectMask = aspectFlags;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		VkImageView imageView;
		if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image view!");
		}
		return imageView;
	}

//------<IMAGE VIEWS>------
//------<GRAPHICS PIPELINE>------
	//take buffer with bytecode and create a VkShaderModule
	VkShaderModule createShaderModule(const std::vector<char>& code) {
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS){
			throw std::runtime_error("failed to create shader module!");
		}

		//free(code);
		return shaderModule;
	}


//------<GRAPHICS PIPELINE>------
//------<COMMAND POOL>------
	void createCommandPool(){
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
		poolInfo.flags = 0; // Optional	

		if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS){
			throw std::runtime_error("failed to create command pool!");
		}
	}
//------</COMMAND POOL>------
//------<BUFFERS>------
	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory){
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create buffer!");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate buffer memory!");
		}

		vkBindBufferMemory(device, buffer, bufferMemory, 0);
		
	}

	VkCommandBuffer beginSingleTimeCommands() {
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		if(vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) != VK_SUCCESS){
			throw std::runtime_error("failed to allocate single time command buffer");	
		}

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		if(vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS){
			throw std::runtime_error("failed to begin single time command buffer!");	
		}

		return commandBuffer;
	}

	void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);

		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	}

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferCopy copyRegion{};
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		endSingleTimeCommands(commandBuffer);
	}

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}


	void createVertexBuffer(){
		VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

		//only use host(cpu)-visible staging buffer as temporary buffer and use device(gpu) local one as the actual vertex buffer
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		//map buffer memory into cpu accessible memory, copy, unmap
		//pointer to mapped memory
		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, vertices.data(), (size_t) bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexB.buffer, vertexB.mem);

		copyBuffer(stagingBuffer, vertexB.buffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void createIndexBuffer() {
		VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
	
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
	
		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, indices.data(), (size_t) bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);
	
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexB.buffer, indexB.mem);
	
		copyBuffer(stagingBuffer, indexB.buffer, bufferSize);
	
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

//------</BUFFERS>------
//------<IMAGE TEXTURE>------
void createMotifResources() {
		//motif image size
		int motifWidth, motifHeight, motifChannels;
		stbi_uc* pixels = stbi_load("textures/motif.png", &motifWidth, &motifHeight, &motifChannels, STBI_rgb_alpha);
		VkDeviceSize imageSize = motifWidth * motifHeight * 4;

		if (!pixels) {
			throw std::runtime_error("failed to load motif image!");
		}

		//copy pixels to gpu
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		vkUnmapMemory(device, stagingBufferMemory);

		stbi_image_free(pixels);

		//image
		createImage(motifWidth, motifHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, motif.image, motif.mem);

		transitionImageLayout(motif.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, 1);
			copyBufferToImage(stagingBuffer, motif.image, static_cast<uint32_t>(motifWidth), static_cast<uint32_t>(motifHeight));
		transitionImageLayout(motif.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, 1);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);

		//view
		motif.view = createImageView(motif.image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);

		//sampler
		createSampler(motif.sampler);
	}


	void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		//relevant to multisampling
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image!");
		}

		//alocate memory
		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(device, image, imageMemory, 0);
	}

	
	void transitionImageLayout(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, VkImageAspectFlags aspectMask, uint32_t layerCount) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		barrier.subresourceRange.aspectMask = aspectMask;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = layerCount;

		//specify which operations must happen before barrier/must wait on barrier.
		//depends on old-/new- Layout
		VkPipelineStageFlags sourceStage;
		VkPipelineStageFlags destinationStage;
		
		//GENERAL -> TRANSFER_DST
		if(oldLayout == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL){
			barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		//PRESENT_SRC_KHR -> TRANSFER_SRC
		else if(oldLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL){
			barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		//TRANSFER_DST -> GENERAL
		else if(oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL){
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		//TRANSFER_DST -> PRESENT_SRC_KHR
		else if(oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR){
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		//TRANSFER_DST -> SHADER_READ
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		//TRANSFER_SRC -> PRESENT_SRC_KHR
		else if(oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR){
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		//UNDEF -> TRANSFER_DST
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		//UNDEF -> TRANSFER_SRC
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else {
			throw std::invalid_argument("unsupported layout transition!");
		}

		vkCmdPipelineBarrier(
			commandBuffer,
			sourceStage, destinationStage,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);

		endSingleTimeCommands(commandBuffer);
	}


	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferImageCopy region{};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;
		region.imageOffset = {0, 0, 0};
		region.imageExtent = {
			width,
			height,
			1
		};

		vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		endSingleTimeCommands(commandBuffer);
	}

	void createSampler(VkSampler& sampler){
		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		//regarding under/over sampling
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		//what to do with texture on axis when going beyond image dimensions (not gonna sample outside)
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.anisotropyEnable = VK_TRUE;
		samplerInfo.maxAnisotropy = 16.0f;
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		//use normalised [0,1]
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		//for filtering
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		//for mipmap
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 0.0f;

		if (vkCreateSampler(device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture sampler!");
		}
	}

//------</IMAGE TEXTURE>------
//------<DEPTH>------
	//from candidate formats, ordered in preference, return first one
	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
		for(VkFormat format : candidates){
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

			if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
				return format;
			} else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
				return format;
			}
		}

		throw std::runtime_error("failed to find supported format!");
	}

	//3 candidate formats, return optimal supported
	VkFormat findDepthFormat() {
		return findSupportedFormat(
			{VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
			VK_IMAGE_TILING_OPTIMAL,
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
		);
	}

	//check if chosen depth format contains a stencil component
	bool hasStencilComponent(VkFormat format) {
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
	}

	void createDepthResources(){
		VkFormat depthFormat = findDepthFormat();
		
		//create image
		createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depth.image, depth.mem);
		//create image view
		depth.view = createImageView(depth.image, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

	}


//------</DEPTH>------
//------<DUMP>------

	bool checkBlitSupport(VkFormat format){
		// Check blit support for source and destination
		VkFormatProperties formatProps;

		// Check if the device supports blitting from optimal images (the swapchain images are in optimal format)
		vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProps);
		if (!(formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT)) {
			std::cerr << "Device does not support blitting from optimal tiled images, using copy instead of blit!" << std::endl;
			return false;
		}

		// Check if the device supports blitting to linear images 
		vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProps);
		if (!(formatProps.linearTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
			std::cerr << "Device does not support blitting to linear tiled images, using copy instead of blit!" << std::endl;
			return false;
		}

		return true;
	}

	//srcLayout = srcOldLayout = srcNewLayout
	void blit(VkImage& srcImage, VkImageLayout srcLayout, VkFormat srcFormat, uint32_t width, uint32_t height, VkImage& dstImage, VkImageLayout dstOldLayout, VkImageLayout dstNewLayout, VkImageAspectFlags aspectMask, uint32_t layerCount){
		// Note that vkCmdBlitImage (if supported) will also do format conversions if the swapchain color format would differ
		bool supportsBlit = checkBlitSupport(srcFormat);
	
		// Transition swapchain image to  transfer source layout
		if(srcLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL){
			transitionImageLayout(srcImage, srcLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, aspectMask, layerCount);
		}
		// Transition destination image to transfer destination layout
		if(dstOldLayout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL){
			transitionImageLayout(dstImage, dstOldLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, aspectMask, layerCount);
		}

		// Do the actual blit from the src image to our host visible destination image
		//create blit/copy command buffer
		VkCommandBuffer copyCmd = beginSingleTimeCommands();

		// If source and destination support blit we'll blit as this also does automatic format conversion (e.g. from BGR to RGB)
		if (supportsBlit){
			// Define the region to blit (we will blit the whole swapchain image)
			VkOffset3D blitSize;
			blitSize.x = WIDTH;
			blitSize.y = HEIGHT;
			blitSize.z = 1;
			VkImageBlit imageBlitRegion{};
			imageBlitRegion.srcSubresource.aspectMask = aspectMask;
			imageBlitRegion.srcSubresource.layerCount = layerCount;
			imageBlitRegion.srcOffsets[1] = blitSize;
			imageBlitRegion.dstSubresource.aspectMask = aspectMask;
			imageBlitRegion.dstSubresource.layerCount = layerCount;
			imageBlitRegion.dstOffsets[1] = blitSize;

			// Issue the blit command
			vkCmdBlitImage(
				copyCmd,
				srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1,
				&imageBlitRegion,
				VK_FILTER_NEAREST);
		}
		else{
			// Otherwise use image copy (requires us to manually flip components)
			VkImageCopy imageCopyRegion{};
			imageCopyRegion.srcSubresource.aspectMask = aspectMask;
			imageCopyRegion.srcSubresource.layerCount = layerCount;
			imageCopyRegion.dstSubresource.aspectMask = aspectMask;
			imageCopyRegion.dstSubresource.layerCount = layerCount;
			imageCopyRegion.extent.width = width;
			imageCopyRegion.extent.height = height;
			imageCopyRegion.extent.depth = 1;

			// Issue the copy command
			vkCmdCopyImage(
				copyCmd,
				srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1,
				&imageCopyRegion);
		}

		endSingleTimeCommands(copyCmd);

	
		// Transition back the src image after the blit is done
		if(srcLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL){
			transitionImageLayout(srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, srcLayout, aspectMask, layerCount);
		}
		// Transition destination image to new layout
		if(dstNewLayout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL){
			transitionImageLayout(dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, dstNewLayout, aspectMask, layerCount);
		}
	}


	void saveOutputColorTexture(const std::string& path, VkImage& srcImage, VkImageLayout srcLayout, VkFormat srcFormat, uint32_t width, uint32_t height){

		// Create the linear tiled destination image to copy to and to read the memory from
		// Note that vkCmdBlitImage (if supported) will also do format conversions if the swapchain color format would differ
		VkImage dstImage;
		VkDeviceMemory dstImageMemory;
		createImage(width, height, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_LINEAR, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, dstImage, dstImageMemory);

		// Transition destination image to general layout, which is the required layout for mapping the image memory later on

		blit(srcImage, srcLayout, srcFormat, width, height, dstImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT, 1);

	// Get layout of the image (including row pitch)
		VkImageSubresource subResource{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
		VkSubresourceLayout subResourceLayout;
		vkGetImageSubresourceLayout(device, dstImage, &subResource, &subResourceLayout);

		// Map image memory so we can start copying from it
		const char* data;
		vkMapMemory(device, dstImageMemory, 0, VK_WHOLE_SIZE, 0, (void**)&data);
		data += subResourceLayout.offset;

		std::ofstream file(path, std::ios::out | std::ios::binary);

		// ppm header
		file << "P6\n" << width << "\n" << height << "\n" << 255 << "\n";

		// If source is BGR (destination is always RGB) and we can't use blit (which does automatic conversion), we'll have to manually swizzle color components
		bool colorSwizzle = false;
		// Check if source is BGR
		// Note: Not complete, only contains most common and basic BGR surface formats for demonstation purposes
		bool supportsBlit = checkBlitSupport(srcFormat);
		if (!supportsBlit)
		{
			std::vector<VkFormat> formatsBGR = { VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SNORM };
			colorSwizzle = (std::find(formatsBGR.begin(), formatsBGR.end(), srcFormat) != formatsBGR.end());
		}

		auto image_size = height * width;


		for (uint32_t y = 0; y < height; y++)
		{
			unsigned int *row = (unsigned int*)data;
			for (uint32_t x = 0; x < width; x++)
			{
				if (colorSwizzle)
				{
					file.write((char*)row + 2, 1);
					file.write((char*)row + 1, 1);
					file.write((char*)row, 1);
				}
				else
				{
					file.write((char*)row, 3);
				}
				row++;
			}
			data += subResourceLayout.rowPitch;
		}

		file.close();


		// Clean up resources
		vkUnmapMemory(device, dstImageMemory);
		vkFreeMemory(device, dstImageMemory, nullptr);
		vkDestroyImage(device, dstImage, nullptr);

	}


//------</DUMP>------
//------<CLEANUP>------
/*
*/
	void cleanupBase(){

		//motif Texture
		vkDestroySampler(device, motif.sampler, nullptr);
		vkDestroyImageView(device, motif.view, nullptr);
		vkDestroyImage(device, motif.image, nullptr);
		vkFreeMemory(device, motif.mem, nullptr);


		//index Buffer
		vkDestroyBuffer(device, indexB.buffer, nullptr);
		vkFreeMemory(device, indexB.mem, nullptr);
		//vertex Buffer
		vkDestroyBuffer(device, vertexB.buffer, nullptr);
		vkFreeMemory(device, vertexB.mem, nullptr);

		//command pool
		vkDestroyCommandPool(device, commandPool, nullptr);
		//logdev
		vkDestroyDevice(device, nullptr);
		//debug
		if(enableValidationLayers){
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}
		//window surface
		vkDestroySurfaceKHR(instance, surface, nullptr);
		//instance
		vkDestroyInstance(instance, nullptr);
	
		glfwDestroyWindow(window);

		glfwTerminate();

	}





//------</CLEANUP>------


};
//------</CLASS>------
