#pragma once
#include "../opgraph.h"
#include "../tensor.h"
#include <vector>

namespace ten {

using KernelFn = void (*)(float **, int);

class Builder {
public:
	TensorLayout matmul(TensorLayout A, TensorLayout B);
	TensorLayout bias_add(TensorLayout x, TensorLayout bias);
	TensorLayout relu(TensorLayout x);
	TensorLayout add(TensorLayout A, TensorLayout B);
	TensorLayout transpose(TensorLayout A);

	KernelFn compile();

	Builder() = default;

private:
	std::vector<OpNode> nodes;

	TensorLayout add_node(Op op, std::vector<TensorLayout> inputs,
						  TensorLayout output);
};

} // namespace ten