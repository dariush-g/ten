#pragma once

#include <tensor.hpp>

namespace ten {

enum Op {
	MATMUL,
	BIAS_ADD,
	RELU,
	TRANSPOSE,
	ADD,
	MUL,
};

struct OpNode {
	Op op;
	std::vector<TensorLayout> inputs;
	TensorLayout output;
	std::vector<size_t> consumers;
	size_t id;
};

} // namespace ten