#pragma once

#include "../opgraph.h"
#include "ir.h"
#include <vector>

namespace ten {
LoopNest lower_matmul(const OpNode &node);
LoopNest lower_relu(const OpNode &node);
LoopNest lower_bias_add(const OpNode &node);
std::vector<LoopNest> lower(const std::vector<OpNode> &nodes);
} // namespace ten