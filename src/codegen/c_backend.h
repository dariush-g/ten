#pragma once

#include "../ir/ir.h"
#include "../opgraph.h"
#include "tree.h"
#include <string>
#include <unordered_map>
#include <vector>

std::string emit_stmt(const ten::codegen::StmtPtr& stmt);

namespace ten::codegen
{
    std::pair<std::string, std::vector<std::string>> emit_c(const std::vector<LoopNest>& nests, std::string name);
} // namespace ten::codegen
