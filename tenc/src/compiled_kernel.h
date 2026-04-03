#pragma once

#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.h"
#include "types.h"

namespace ten
{
   class CompiledKernel
   {
      KernelFn fn_;
      std::vector<std::string> tensor_order_;
      std::unordered_map<std::string, TensorLayout> layouts_;

      std::vector<std::vector<float>> owned_buffers_;
      std::unordered_map<std::string, float*> ptrs_;
      bool has_run_ = false;

   public:
      CompiledKernel(const KernelFn fn, std::vector<std::string> tensor_order,
                     std::unordered_map<std::string, TensorLayout> layouts)
         : fn_(fn), tensor_order_(std::move(tensor_order)),
           layouts_(std::move(layouts))
      {
      }

      void operator()(const std::unordered_map<std::string, float*>& named)
      {
         owned_buffers_.clear();
         ptrs_.clear();
         std::vector<float*> ptrs(tensor_order_.size());

         for (size_t i = 0; i < tensor_order_.size(); i++)
         {
            auto& name = tensor_order_[i];
            if (auto it = named.find(name); it != named.end())
            {
               ptrs[i] = it->second;
            }
            else
            {
               auto& layout = layouts_.at(name);
               int n_elems = 1;
               for (const int d : layout.shape)
                  n_elems *= d;
               owned_buffers_.emplace_back(n_elems, 0.0f);
               ptrs[i] = owned_buffers_.back().data();
            }
            ptrs_[name] = ptrs[i];
         }

         fn_(ptrs.data(), static_cast<int>(tensor_order_.size()));
         has_run_ = true;
      }

      [[nodiscard]] TensorView get(const TensorLayout& layout) const
      {
         if (!has_run_)
            throw std::runtime_error("kernel has not been run yet");
         return {ptrs_.at(layout.name), layout.shape};
      }

      [[nodiscard]] const std::vector<std::string>& tensor_order() const
      {
         return tensor_order_;
      }
   };
} // namespace ten
