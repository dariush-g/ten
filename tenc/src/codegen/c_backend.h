#pragma once

#include "../ir/ir.h"
#include "../opgraph.h"
#include "tree.h"
#include <string>
#include <unordered_map>
#include <vector>

std::string emit_stmt(ten::codegen::StmtPtr stmt);
namespace ten::codegen {
std::string emit_c(const std::vector<StmtPtr> &stmts);
} // namespace ten::codegen