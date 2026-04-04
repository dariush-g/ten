#include "scheduler.h"

namespace ten::scheduler
{
    LoopNest Scheduler::tile(LoopNest nest, const std::string& index, const int factor)
    {
        nest.tiled[index] = {index + "_outer", index + "_inner"};
        for (size_t i = 0; i < nest.indices.size(); ++i)
        {
            if (nest.indices[i].name == index)
            {
                const auto idx = nest.indices[i];
                nest.indices.erase(nest.indices.begin() + i);
                nest.indices.push_back({
                    .name = index + "_outer",
                    .extent = idx.extent / factor,
                    .tile_factor = factor,
                    .is_reduction = idx.is_reduction,
                });
                nest.indices.push_back({
                    .name = index + "_inner",
                    .extent = factor,
                    .tile_factor = factor,
                    .is_reduction = idx.is_reduction,
                });
            }
        }

        for (size_t i = 0; i < nest.order.size(); ++i)
        {
            for (size_t i = 0; i < nest.order.size(); ++i)
            {
                if (nest.order[i] == index)
                {
                    nest.order[i] = index + "_outer";
                    nest.order.insert(nest.order.begin() + i + 1, index + "_inner");
                    break;
                }
            }
        }

        return nest;
    }
}
