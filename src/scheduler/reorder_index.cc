
#include "scheduler.h"
#include <cassert>

namespace ten::scheduler
{
    LoopNest Scheduler::reorder(LoopNest nest, std::vector<std::string> order)
    {
        assert(order.size() == nest.order.size());
        nest.order = std::move(order);
        return nest;
    }
}
