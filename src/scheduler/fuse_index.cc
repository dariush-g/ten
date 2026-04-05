#include "scheduler.h"

namespace ten::scheduler
{
    std::vector<LoopNest> Scheduler::fuse(const std::vector<LoopNest>& nests)
    {
        std::vector<LoopNest> result;

        if (nests.empty())
            return result;
        result.push_back(nests[0]);

        for (size_t i = 1; i < nests.size(); i++)
        {
            auto& prev = result.back();
            auto& curr = nests[i];

            bool elementwise = true;
            for (const auto& idx : curr.indices)
                if (idx.is_reduction)
                {
                    elementwise = false;
                    break;
                }

            if (elementwise /* && same dimensions as prev output */)
            {
                prev.epilogue.push_back(curr.body);
                for (auto& [name, layout] : curr.tensors)
                    prev.tensors[name] = layout;
            }
            else
            {
                result.push_back(curr);
            }
        }


        return result;
    }
}
