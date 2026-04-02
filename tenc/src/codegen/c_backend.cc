#include "c_backend.h"

std::unordered_map<std::string, int>
build_tensor_index(const ten::LoopNest &nest) {
	std::unordered_map<std::string, int> idx;
	int i = 0;
	for (auto &[name, layout] : nest.tensors) {
		idx[name] = i++;
	}
	return idx;
}

std::string emit_index(const ten::TensorAccess &access,
					   const ten::LoopNest &nest) {}

std::string emit_compute(const ten::Compute &compute, const ten::LoopNest &nest,
						 const std::string &indent) {}

std::string emit_nest(const ten::LoopNest &nest,
					  const std::unordered_map<std::string, int> &tensor_idx) {}

namespace ten::codegen {
std::string emit_c(const std::vector<ten::LoopNest> &nests) {}
} // namespace ten::codegen