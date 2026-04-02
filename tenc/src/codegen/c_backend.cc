#include "c_backend.h"
#include <sstream>

std::string emit_stmt(ten::codegen::StmtPtr stmt) { return (*stmt).emit_c(0); }

namespace ten::codegen {
std::string emit_c(const std::vector<StmtPtr> &stmts) {
	std::stringstream str;
	for (auto stmt : stmts) {
		str << emit_stmt(stmt);
	}
	return str.str();
}
} // namespace ten::codegen