#include "scheduler.h"

namespace ten::scheduler
{
    LoopNest Scheduler::tile(LoopNest nest, const std::string& index, int factor)
    {
        nest.tiled[index] = {index + "_outer", index + "_inner"};
        for (auto& idx : nest.indices)
        {
            if (idx.name == index)
            {
                idx.tile_factor = factor;
            }
        }

        return nest;
    }
}
