#pragma once

#include "../builder/builder.hpp"
#include "../opgraph.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace ten {

struct Index {
	std::string name;
	size_t extent;

	size_t tile_factor;
	bool is_reduction;
};

struct TensorAccess {
	std::string tensor_name;
	std::vector<std::string> indices;
	bool is_output;
};

struct Compute {
	TensorAccess output;
	std::vector<TensorAccess> inputs;
	Op op;
};

struct LoopNest {
	std::vector<Index> indices;
	std::vector<std::string> order;

	Compute body;
	std::unordered_map<std::string, TensorLayout> tensors;
	std::unordered_map<std::string, std::pair<std::string, std::string>> tiled;

	std::vector<Compute> epilogue;
};

} // namespace ten