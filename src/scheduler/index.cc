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


    	if (!(flags & NO_TILE)) {
    		for (auto & nest : nests) {
    			for (const auto indices = nest.indices; const auto& idx : indices)
    				if (!idx.is_reduction)
    					if (const int f = pick_tile_factor(idx.extent); f > 1)
    						nest = tile(nest, idx.name, f);

    			std::vector<std::string> outers, inners, reductions;
    			for (auto& name : nest.order) {
    				if (name.ends_with("_outer")) outers.push_back(name);
    				else if (name.ends_with("_inner")) inners.push_back(name);
    				else reductions.push_back(name);
    			}
    			std::vector<std::string> new_order;
    			new_order.insert(new_order.end(), outers.begin(), outers.end());
    			new_order.insert(new_order.end(), reductions.begin(), reductions.end());
    			new_order.insert(new_order.end(), inners.begin(), inners.end());
    			nest = reorder(nest, new_order);
    		}
    	}
    }
}
