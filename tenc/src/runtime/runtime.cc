// src/runtime/runtime.cc
#include "runtime.h"
#include <dlfcn.h>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace ten::runtime {
    static std::unordered_map<std::string, CachedKernel> cache;

    const CachedKernel &get_or_compile(
        const std::string &key,
        const std::string &code,
        const std::vector<std::string> &tensor_order
    ) {
        if (auto it = cache.find(key); it != cache.end())
            return it->second;

        std::string src = "/tmp/ten_" + key + ".cc";
        std::string lib = "/tmp/ten_" + key + ".so";

        std::ofstream f(src);
        if (!f) throw std::runtime_error("failed to write: " + src);
        f << code;
        f.close();

#ifdef TEN_DEBUG
        std::string cmd = "clang++ -O0 -shared -fPIC -o " + lib + " " + src;
#else
        std::string cmd = "clang++ -O3 -march=native -ffast-math"
                          " -shared -fPIC -o " + lib + " " + src;
#endif

        if (system(cmd.c_str()) != 0)
            throw std::runtime_error("compilation failed: " + src);

        void *handle = dlopen(lib.c_str(), RTLD_NOW);
        if (!handle)
            throw std::runtime_error(std::string("dlopen: ") + dlerror());

        void *sym = dlsym(handle, "kernel");
        if (!sym) {
            dlclose(handle);
            throw std::runtime_error(std::string("dlsym: ") + dlerror());
        }

        CachedKernel kernel;
        kernel.fn = reinterpret_cast<KernelFn>(sym);
        kernel.lib_handle = handle;
        kernel.tensor_order = tensor_order;

        cache[key] = kernel;
        return cache[key];
    }
} // namespace ten::runtime
