#include "lower.h"

namespace ten::codegen {
std::vector<Stmt> lower_index(const TensorAccess &access,
							  const LoopNest &nest) {
	auto &layout = nest.tensors.at(access.tensor_name);
}

std::vector<Stmt> lower_compute(const Compute &compute, const LoopNest &nest) {}

std::vector<Stmt> lower_nest(const LoopNest &nest) {
	auto stmts = lower_compute(nest.body, nest);

	return stmts;
}
} // namespace ten::codegen