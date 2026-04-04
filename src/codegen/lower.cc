#include "lower.h"

namespace ten::codegen
{
	std::vector<StmtPtr> lower_index(const TensorAccess& access,
	                                 const LoopNest& nest)
	{
		auto& layout = nest.tensors.at(access.tensor_name);
		return {};
	}

	ExprPtr make_index_expr(const std::string& idx_name, const LoopNest& nest)
	{
		if (nest.tiled.contains(idx_name))
		{
			auto& [outer, inner] = nest.tiled.at(idx_name);

			int factor = -1;
			for (auto& idx : nest.indices)
				if (idx.name == inner)
				{
					factor = idx.tile_factor;
					break;
				}
			if (factor == -1)
				throw std::runtime_error("tiled index " + idx_name +
					" not found in nest.indices");

			return std::make_shared<BinOp>(
				"+",
				std::make_shared<BinOp>("*", std::make_shared<Var>(outer),
				                        std::make_shared<IntLit>(factor)),
				std::make_shared<Var>(inner));
		}
		return std::make_shared<Var>(idx_name);
	}

	ExprPtr make_array_access(const TensorAccess& access, const LoopNest& nest)
	{
		auto& layout = nest.tensors.at(access.tensor_name);

		ExprPtr flat_idx = nullptr;

		for (int i = 0; i < static_cast<int>(access.indices.size()); i++)
		{
			int stride = 1;
			for (int d = i + 1; d < layout.rank(); d++)
				stride *= layout.dim(d);

			auto idx_expr = make_index_expr(access.indices[i], nest);
			ExprPtr term =
				stride == 1
					? idx_expr
					: std::make_shared<BinOp>(
						"*", idx_expr, std::make_shared<IntLit>(stride));

			flat_idx = flat_idx == nullptr
				           ? term
				           : std::make_shared<BinOp>("+", flat_idx, term);
		}
		if (!flat_idx)
			flat_idx = std::make_shared<IntLit>(0);
		return std::make_shared<ArrayAccess>(access.tensor_name, flat_idx);
	}

	StmtPtr lower_matmul(const Compute& compute, const LoopNest& nest)
	{
		auto lhs = make_array_access(compute.output, nest);
		auto rhs =
			std::make_shared<BinOp>("*", make_array_access(compute.inputs[0], nest),
			                        make_array_access(compute.inputs[1], nest));

		return {std::make_shared<Assign>(lhs, "+=", rhs)};
	}

	StmtPtr lower_ba(const Compute& compute, const LoopNest& nest)
	{
		auto lhs = make_array_access(compute.output, nest);
		auto rhs =
			std::make_shared<BinOp>("+", make_array_access(compute.inputs[0], nest),
			                        make_array_access(compute.inputs[1], nest));

		return {std::make_shared<Assign>(lhs, "=", rhs)};
	}

	StmtPtr lower_relu(const Compute& compute, const LoopNest& nest)
	{
		auto x_access = make_array_access(compute.inputs[0], nest);
		auto zero = std::make_shared<FloatLit>(0.0f);

		auto cond = std::make_shared<BinOp>(">", x_access, zero);
		auto then_expr = make_array_access(compute.inputs[0], nest);
		auto else_expr = std::make_shared<FloatLit>(0.0f);

		auto relu_expr = std::make_shared<Ternary>(cond, then_expr, else_expr);
		auto lhs = make_array_access(compute.output, nest);

		return {std::make_shared<Assign>(lhs, "=", relu_expr)};
	}

	std::vector<StmtPtr> lower_compute(const Compute& compute,
	                                   const LoopNest& nest)
	{
		auto stmts = std::vector<StmtPtr>();

		switch (compute.op)
		{
		case Op::MATMUL:
			{
				const auto mm = lower_matmul(compute, nest);
				stmts.push_back(mm);

				break;
			}
		case Op::BIAS_ADD:
			{
				const auto ba = lower_ba(compute, nest);
				stmts.push_back(ba);

				break;
			}
		case Op::RELU:
			{
				const auto rl = lower_relu(compute, nest);
				stmts.push_back(rl);
				break;
			}
		default:
			break;
		}

		return stmts;
	}

	std::shared_ptr<Function> lower_nest(const LoopNest& nest,
	                                     const std::unordered_map<std::string, int>& tensor_idx)
	{
		std::vector<StmtPtr> stmts;

		if (nest.order.empty())
			return nullptr;

		auto find_index = [&](const std::string& name) -> const Index&
		{
			for (auto& i : nest.indices)
				if (i.name == name)
					return i;
			throw std::runtime_error("index not found: " + name);
		};

		std::vector<std::shared_ptr<ForLoop>> loops;
		for (auto& idx_name : nest.order)
		{
			auto& index = find_index(idx_name);
			loops.push_back(std::make_shared<ForLoop>(idx_name, 0, index.extent));
		}

		for (auto& stmt : lower_compute(nest.body, nest))
			loops.back()->body.push_back(stmt);

		for (auto& ep : nest.epilogue)
			for (auto& stmt : lower_compute(ep, nest))
				loops.back()->body.push_back(stmt);

		for (int i = static_cast<int>(loops.size()) - 1; i > 0; i--)
			loops[i - 1]->body.push_back(loops[i]);

		stmts.push_back(loops[0]);

		return std::make_shared<Function>(Function("kernel", stmts));
	}
} // namespace ten::codegen
