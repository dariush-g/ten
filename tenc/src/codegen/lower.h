#pragma once

#include "../ir/ir.h"
#include "tree.h"
#include <vector>

namespace ten::codegen {
std::vector<Stmt> lower_index(const TensorAccess &access, const LoopNest &nest);

std::vector<Stmt> lower_compute(const Compute &compute, const LoopNest &nest);

std::vector<Stmt> lower_nest(const LoopNest &nest);
} // namespace ten::codegen