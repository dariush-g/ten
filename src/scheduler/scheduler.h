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

        static void run(std::vector<LoopNest>& nests, unsigned flags = NONE);

        [[nodiscard]] static LoopNest reorder(LoopNest nest, std::vector<std::string> order);
        [[nodiscard]] static std::vector<LoopNest> fuse(const std::vector<LoopNest>& nests);
        [[nodiscard]] static LoopNest tile(LoopNest nest, const std::string& index, int factor);
    };
}

