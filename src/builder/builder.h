#pragma once

#include "../compiled_kernel.h"
#include "../opgraph.h"
#include "../tensor.h"
#include <vector>
#include <string>
#include "flags.h"

namespace ten
{
	class Builder
	{
	public:
		TensorLayout matmul(const TensorLayout& A, const TensorLayout& B);

		TensorLayout bias_add(const TensorLayout& x, const TensorLayout& bias);

		TensorLayout relu(const TensorLayout& x);

		TensorLayout add(const TensorLayout& A, const TensorLayout& B);

		TensorLayout transpose(const TensorLayout& A);

		[[nodiscard]] std::string make_cache_key() const;

		[[nodiscard]] CompiledKernel compile(unsigned flags = NONE) const;

		Builder() = default;

	private:
		std::vector<OpNode> nodes;

		TensorLayout add_node(Op op, const std::vector<TensorLayout>& inputs,
		                      TensorLayout output);
	};
} // namespace ten
