#pragma once

#include <string>
#include <vector>
#include "../types.h"

namespace ten::runtime {
    struct CachedKernel {
        KernelFn fn;
        void *lib_handle;
        std::vector<std::string> tensor_order;
    };

    const CachedKernel &get_or_compile(
        const std::string &key,
        const std::string &code,
        const std::vector<std::string> &tensor_order
    );
}
