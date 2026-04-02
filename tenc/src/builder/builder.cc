#include "builder.h"
#include "../ir/lower.h"
#include <stdexcept>

namespace ten {

TensorLayout Builder::add_node(Op op, std::vector<TensorLayout> inputs,
							   TensorLayout output) {
	size_t id = nodes.size();

	for (auto &input : inputs) {
		for (auto &node : nodes) {
			if (node.output.name == input.name) {
				node.consumers.push_back(id);
			}
		}
	}

	nodes.push_back(OpNode{op, inputs, output, {}, id});
	return output;
}

TensorLayout Builder::matmul(TensorLayout A, TensorLayout B) {
	if (A.rank() != 2 || B.rank() != 2)
		throw std::invalid_argument("matmul requires 2D tensors");
	if (A.dim(1) != B.dim(0))
		throw std::invalid_argument("matmul shape mismatch");
	if (A.dtype != B.dtype)
		throw std::invalid_argument("matmul dtype mismatch");

	TensorLayout C({A.dim(0), B.dim(1)}, A.dtype,
				   "t" + std::to_string(nodes.size()));
	return add_node(Op::MATMUL, {A, B}, C);
}

TensorLayout Builder::bias_add(TensorLayout x, TensorLayout bias) {
	if (bias.rank() != 1)
		throw std::invalid_argument("bias must be 1D");
	if (x.dim(x.rank() - 1) != bias.dim(0))
		throw std::invalid_argument("bias size must match last dim of x");

	TensorLayout out = x;
	out.name = "t" + std::to_string(nodes.size());
	return add_node(Op::BIAS_ADD, {x, bias}, out);
}

TensorLayout Builder::relu(TensorLayout x) {
	TensorLayout out = x;
	out.name = "t" + std::to_string(nodes.size());
	return add_node(Op::RELU, {x}, out);
}

TensorLayout Builder::add(TensorLayout A, TensorLayout B) {
	if (A.shape != B.shape)
		throw std::invalid_argument("add shape mismatch");
	if (A.dtype != B.dtype)
		throw std::invalid_argument("add dtype mismatch");

	TensorLayout out = A;
	out.name = "t" + std::to_string(nodes.size());
	return add_node(Op::ADD, {A, B}, out);
}

TensorLayout Builder::transpose(TensorLayout A) {
	if (A.rank() != 2)
		throw std::invalid_argument("transpose requires 2D tensor");

	TensorLayout out({A.dim(1), A.dim(0)}, A.dtype,
					 "t" + std::to_string(nodes.size()));
	return add_node(Op::TRANSPOSE, {A}, out);
}

KernelFn Builder::compile() {
	auto loops = lower(nodes);

	return nullptr;
}

} // namespace ten