#include "c_backend.h"
#include "lower.h"
#include <sstream>

std::string emit_stmt(const ten::codegen::StmtPtr& stmt) { return stmt->emit_c(0); }

namespace ten::codegen
{
	std::pair<std::string, std::vector<std::string>> emit_c(const std::vector<LoopNest>& nests, std::string f_name)
	{
		std::ostringstream ss;
		ss << "#include <stdint.h>\n#include <stdio.h>\n";
		ss << "extern \"C\" void " + f_name + "(float** tensors, int n) {\n";

		std::vector<std::string> tensor_order;
		std::unordered_map<std::string, TensorLayout> all_tensors;

		for (auto& nest : nests)
		{
			for (auto& [name, layout] : nest.tensors)
			{
				if (!all_tensors.contains(name))
				{
					tensor_order.push_back(name);
					all_tensors[name] = layout;
				}
			}
		}

		std::unordered_map<std::string, int> tensor_idx;
		for (int i = 0; i < static_cast<int>(tensor_order.size()); i++)
			tensor_idx[tensor_order[i]] = i;

		for (auto& name : tensor_order)
		{
			auto& layout = all_tensors[name];
			ss << "    " << dtype_str(layout.dtype) << "* " << name << " = ("
				<< dtype_str(layout.dtype) << "*)tensors[" << tensor_idx[name]
				<< "];\n";
		}
		ss << "\n";

		for (auto& nest : nests)
		{
			for (auto stmts = lower_nest(nest)->body; const auto& stmt : stmts)
				ss << stmt->emit_c(1);
			ss << "\n";
		}


		ss << "}\n";
		return {ss.str(), tensor_order};
	}
} // namespace ten::codegen
