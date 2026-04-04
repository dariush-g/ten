#pragma once
#include <utility>
#include <vector>
#include <string>

#include "../tensor.h"
#include "../ir/ir.h"

namespace ten::scheduler
{
    class Scheduler
    {
    public:
        Scheduler() = default;

        [[nodiscard]] std::vector<LoopNest> run(std::vector<LoopNest> nests);

        [[nodiscard]] static LoopNest reorder(LoopNest nest, std::vector<std::string> order);
        [[nodiscard]] static std::vector<LoopNest> fuse(std::vector<LoopNest>& nests);
        [[nodiscard]] static LoopNest tile(LoopNest nest, const std::string& index, int factor);
    };
}

