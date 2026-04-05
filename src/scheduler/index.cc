#include <iostream>

#include "scheduler.h"

namespace ten::scheduler
{
    int pick_tile_factor(const int extent)
    {
        for (const int f : {32, 16, 8, 4})
            if (extent % f == 0) return f;
        return 1;
    }

    void Scheduler::run(std::vector<LoopNest>& nests, const unsigned flags)
    {
        if (!(flags & CompileFlags::NO_FUSE))

            nests = fuse(nests);


        if (!(flags & NO_TILE) && !nests.empty())
        {
            for (const auto indices = nests[0].indices; const auto& idx : indices)
                if (!idx.is_reduction)
                    if (const int f = pick_tile_factor(idx.extent); f > 1)
                        nests[0] = tile(nests[0], idx.name, f);
        }
    }
}
