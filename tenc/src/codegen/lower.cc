#include "lower.h"

namespace ten::codegen {
std::vector<StmtPtr> lower_index(const TensorAccess &access,
								 const LoopNest &nest) {
	auto &layout = nest.tensors.at(access.tensor_name);
	return {};
}

std::vector<StmtPtr> lower_compute(const Compute &compute,
								   const LoopNest &nest) {
	return {};
}

std::vector<StmtPtr> lower_nest(const LoopNest &nest) {
	std::vector<StmtPtr> stmts = std::vector<StmtPtr>();
	for (auto index : nest.indices) {
		ForLoop fl(index.name, 0, index.extent);
		stmts.push_back(std::make_shared<ForLoop>(fl));
	}

	for (auto stmt : lower_compute(nest.body, nest)) {
		stmts.push_back(stmt);
	}

	return stmts;
}
} // namespace ten::codegen