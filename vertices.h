static const std::vector<uint16_t> indices = {

	//floor
	0,2,1, 0,3,2, 0,1,2, 0,2,3,

	//cube
	4,6,5, 4,7,6, 4,5,6, 4,6,7,
	4,8,11, 4,11,7, 4,11,8, 4,7,11,
	4,8,9, 4,9,5, 4,9,8, 4,5,9,
	5,9,10, 5,10,6, 5,10,9, 5,6,10,
	6,7,11, 6,11,10, 6,11,7, 6,10,11,
	8,10,9, 8,11,10, 8,9,10, 8,10,11,

	//pyramid
	12,13,16, 12,16,13,
	12,13,14, 12,14,13,
	12,14,15, 12,15,14,
	12,16,15, 12,15,16,
	13,15,14, 13,16,15, 13,14,15, 13,15,16
};


struct Vertex{
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;

	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions(){

		std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

		//pos description
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		//format because its vec2
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		//color description
		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		//texture description
		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

		return attributeDescriptions;
	}
};

static const std::vector<Vertex> vertices = {
	{{-1.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}}, //floor
	{{1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
	{{1.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
	{{-1.0f, -1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},

	{{0.3f, 0.7f, 0.05f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}}, //cube
	{{0.7f, 0.7f, 0.05f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
	{{0.7f, 0.3f, 0.05f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
	{{0.3f, 0.3f, 0.05f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
	{{0.3f, 0.7f, 0.35f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
	{{0.7f, 0.7f, 0.35f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
	{{0.7f, 0.3f, 0.35f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
	{{0.3f, 0.3f, 0.35f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},

	{{0.25f, -0.15f, 0.2f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}}, //pyramid
	{{0.0f, 0.1f, 0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
	{{0.5f, 0.1f, 0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
	{{0.5f, -0.4f, 0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
	{{0.0f, -0.4f, 0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}}
};

/*
*/
