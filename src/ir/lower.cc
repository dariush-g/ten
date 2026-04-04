#include "../opgraph.h"
#include "ir.h"
#include <stdexcept>

namespace ten
{
	LoopNest lower_matmul(const OpNode& node)
	{
		TensorLayout A = node.inputs[0];
		TensorLayout B = node.inputs[1];
		TensorLayout C = node.output;

		int M = A.dim(0);
		int K = A.dim(1);
		int N = B.dim(1);

		Index i;
		i.name = "i";
		i.extent = M;

		Index j;
		j.name = "j";
		j.extent = N;

		Index k;
		k.name = "k";
		k.extent = K;
		k.is_reduction = true;

		LoopNest nest;
		nest.indices = {i, j, k};
		nest.order = {"i", "j", "k"};

		nest.body = Compute{
			.output = TensorAccess{C.name, {"i", "j"}, true},
			.inputs =
			{
				TensorAccess{A.name, {"i", "k"}, false},
				TensorAccess{B.name, {"k", "j"}, false},
			},
			.op = Op::MATMUL
		};

		nest.tensors[A.name] = A;
		nest.tensors[B.name] = B;
		nest.tensors[C.name] = C;

		return nest;
	}

	LoopNest lower_relu(const OpNode& node)
	{
		const TensorLayout x = node.inputs[0];

		const int M = x.dim(0);
		const int N = x.dim(1);

		Index i;
		i.name = "i";
		i.extent = M;

		Index j;
		j.name = "j";
		j.extent = N;

		LoopNest nest;
		nest.indices = {i, j};
		nest.order = {"i", "j"};

		nest.body =
			Compute{
				.output = TensorAccess{node.output.name, {"i", "j"}, true},
				.inputs =
				{
					TensorAccess{x.name, {"i", "j"}, false},
				},
				.op = Op::RELU
			};

		nest.tensors[x.name] = x;
		nest.tensors[node.output.name] = node.output;

		return nest;
	}

	LoopNest lower_bias_add(const OpNode& node)
	{
		TensorLayout x = node.inputs[0];
		TensorLayout bias = node.inputs[1];

		const int M = x.dim(0);
		const int N = x.dim(1);

		Index i;
		i.name = "i";
		i.extent = M;

		Index j;
		j.name = "j";
		j.extent = N;

		LoopNest nest;
		nest.indices = {i, j};
		nest.order = {"i", "j"};

		nest.body =
			Compute{
				.output = TensorAccess{node.output.name, {"i", "j"}, true},
				.inputs =
				{
					TensorAccess{x.name, {"i", "j"}, false},
					TensorAccess{bias.name, {"j"}, false},
				},
				.op = Op::BIAS_ADD
			};

		nest.tensors[x.name] = x;
		nest.tensors[bias.name] = bias;
		nest.tensors[node.output.name] = node.output;

		return nest;
	}

	std::vector<LoopNest> lower(const std::vector<OpNode>& nodes)
	{
		std::vector<LoopNest> nests;

		for (const auto& node : nodes)
		{
			switch (node.op)
			{
			case Op::MATMUL:
				nests.push_back(lower_matmul(node));
				break;
			case Op::RELU:
				nests.push_back(lower_relu(node));
				break;
			case Op::BIAS_ADD:
				nests.push_back(lower_bias_add(node));
				break;
			default:
				throw std::runtime_error("unknown op in lower()");
			}
		}

		return nests;
	}
} // namespace ten
