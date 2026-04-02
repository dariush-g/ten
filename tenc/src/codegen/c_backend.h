#pragma once

#include "../ir/ir.h"
#include "../opgraph.h"
#include <string>
#include <unordered_map>
#include <vector>

std::unordered_map<std::string, int> build_tensor_index(
	const ten::LoopNest
		&nest); // maps tensor name to its index in the tensors array
std::string emit_index(const ten::TensorAccess &access,
					   const ten::LoopNest &nest);
std::string emit_compute(const ten::Compute &compute, const ten::LoopNest &nest,
						 const std::string &indent);
std::string emit_nest(const ten::LoopNest &nest,
					  const std::unordered_map<std::string, int> &tensor_idx);

namespace ten::codegen {
std::string emit_c(const std::vector<ten::LoopNest> &nests);
} // namespace