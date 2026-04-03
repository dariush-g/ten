#include "c_backend.h"
#include "lower.h"
#include <sstream>

std::string emit_stmt(ten::codegen::StmtPtr stmt) { return (*stmt).emit_c(0); }

namespace ten::codegen {
	std::string emit_c(const std::vector<LoopNest> &nests) {
		std::ostringstream ss;
		ss << "#include <stdint.h>\n\n";
		ss << "void kernel(float** tensors, int n) {\n";

		std::vector<std::string> tensor_order;
		std::unordered_map<std::string, TensorLayout> all_tensors;

		for (auto &nest: nests) {
			for (auto &[name, layout]: nest.tensors) {
				if (!all_tensors.count(name)) {
					tensor_order.push_back(name);
					all_tensors[name] = layout;
				}
			}
		}

		std::unordered_map<std::string, int> tensor_idx;
		for (int i = 0; i < (int) tensor_order.size(); i++)
			tensor_idx[tensor_order[i]] = i;

		for (auto &name: tensor_order) {
			auto &layout = all_tensors[name];
			ss << "    " << dtype_str(layout.dtype) << "* " << name << " = ("
					<< dtype_str(layout.dtype) << "*)tensors[" << tensor_idx[name]
					<< "];\n";
		}
		ss << "\n";

		for (auto &nest: nests) {
			auto stmts = lower_nest(nest, tensor_idx);
			for (auto &stmt: stmts)
				ss << stmt->emit_c(1);
			ss << "\n";
		}

		ss << "}\n";
		return ss.str();
	}
} // namespace ten::codegen
