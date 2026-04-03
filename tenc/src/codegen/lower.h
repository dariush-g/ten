#pragma once

#include "../ir/ir.h"
#include "tree.h"
#include <vector>

namespace ten::codegen {
	std::vector<StmtPtr> lower_index(const TensorAccess &access,
	                                 const LoopNest &nest);

	std::vector<StmtPtr> lower_compute(const Compute &compute,
	                                   const LoopNest &nest);

	std::shared_ptr<Function>
	lower_nest(const LoopNest &nest,
	           const std::unordered_map<std::string, int> &tensor_idx);
} // namespace ten::codegen
