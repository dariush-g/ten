#pragma once
#include <unordered_map>
#include <vector>
#include "../tensor.h"
#include "../ir/ir.h"

namespace ten::scheduler
{
    class Scheduler
    {
        std::vector<ten::TensorLayout> layouts;

    public:
        [[nodiscard]] LoopNest reorder(std::vector<LoopNest> nests) const;
    };
}

