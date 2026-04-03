#include "builder.h"
#include "../codegen/c_backend.h"
#include "../codegen/lower.h"
#include "../ir/lower.h"
#include <stdexcept>
#include <unordered_set>

#include "runtime/runtime.h"

namespace ten {
	TensorLayout Builder::add_node(const Op op, const std::vector<TensorLayout> &inputs,
	                               TensorLayout output) {
		const size_t id = nodes.size();

		for (auto &input: inputs) {
			for (auto &node: nodes) {
				if (node.output.name == input.name) {
					node.consumers.push_back(id);
				}
			}
		}

		nodes.push_back(OpNode{op, inputs, output, {}, id});
		return output;
	}

	TensorLayout Builder::matmul(const TensorLayout &A, const TensorLayout &B) {
		if (A.rank() != 2 || B.rank() != 2)
			throw std::invalid_argument("matmul requires 2D tensors");
		if (A.dim(1) != B.dim(0))
			throw std::invalid_argument("matmul shape mismatch");
		if (A.dtype != B.dtype)
			throw std::invalid_argument("matmul dtype mismatch");

		const TensorLayout C({A.dim(0), B.dim(1)}, A.dtype,
		                     "t" + std::to_string(nodes.size()));
		return add_node(Op::MATMUL, {A, B}, C);
	}

	TensorLayout Builder::bias_add(const TensorLayout &x, const TensorLayout &bias) {
		if (bias.rank() != 1)
			throw std::invalid_argument("bias must be 1D");
		if (x.dim(x.rank() - 1) != bias.dim(0))
			throw std::invalid_argument("bias size must match last dim of x");

		TensorLayout out = x;
		out.name = "t" + std::to_string(nodes.size());
		return add_node(Op::BIAS_ADD, {x, bias}, out);
	}

	TensorLayout Builder::relu(const TensorLayout &x) {
		TensorLayout out = x;
		out.name = "t" + std::to_string(nodes.size());
		return add_node(Op::RELU, {x}, out);
	}

	TensorLayout Builder::add(const TensorLayout &A, const TensorLayout &B) {
		if (A.shape != B.shape)
			throw std::invalid_argument("add shape mismatch");
		if (A.dtype != B.dtype)
			throw std::invalid_argument("add dtype mismatch");

		TensorLayout out = A;
		out.name = "t" + std::to_string(nodes.size());
		return add_node(Op::ADD, {A, B}, out);
	}

	TensorLayout Builder::transpose(const TensorLayout &A) {
		if (A.rank() != 2)
			throw std::invalid_argument("transpose requires 2D tensor");

		const TensorLayout out({A.dim(1), A.dim(0)}, A.dtype,
		                       "t" + std::to_string(nodes.size()));
		return add_node(Op::TRANSPOSE, {A}, out);
	}

	std::string Builder::make_cache_key() const {
		std::string key;
		for (auto &node: nodes) {
			switch (node.op) {
				case Op::MATMUL: key += "mm";
					break;
				case Op::BIAS_ADD: key += "ba";
					break;
				case Op::RELU: key += "rl";
					break;
				case Op::ADD: key += "ad";
					break;
				case Op::TRANSPOSE: key += "tr";
					break;
				default: key += "uk";
					break;
			}
			for (auto d: node.output.shape)
				key += "_" + std::to_string(d);
			key += "__";
		}
		return key;
	}

	CompiledKernel Builder::compile() const {
		const auto loops = lower(nodes);

		const auto [code, tensor_order] = codegen::emit_c(loops);

		std::unordered_map<std::string, TensorLayout> all_layouts;
		for (auto &nest : loops)
			for (auto &[name, layout] : nest.tensors)
				if (!all_layouts.contains(name))
					all_layouts[name] = layout;

		const std::string key = make_cache_key();
		auto &cached = ten::runtime::get_or_compile(key, code, tensor_order);
		return CompiledKernel(cached.fn, tensor_order, std::move(all_layouts));
	}
} // namespace ten
