#pragma once

#include "tensor.h"
#include "types.h"
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ten {
	class CompiledKernel {
		KernelFn fn_;
		std::vector<std::string> tensor_order_;
		std::unordered_map<std::string, TensorLayout> layouts_;

	public:
		CompiledKernel(const KernelFn fn, std::vector<std::string> tensor_order,
		               std::unordered_map<std::string, TensorLayout> layouts)
			: fn_(fn), tensor_order_(std::move(tensor_order)),
			  layouts_(std::move(layouts)) {
		}

		void operator()(const std::unordered_map<std::string, float *> &named) const {
			std::vector<std::vector<float> > intermediates;
			std::vector<float *> ptrs(tensor_order_.size());

			for (size_t i = 0; i < tensor_order_.size(); i++) {
				auto &name = tensor_order_[i];
				if (auto it = named.find(name); it != named.end()) {
					ptrs[i] = it->second;
				} else {
					auto &layout = layouts_.at(name);
					int nelems = 1;
					for (const int d: layout.shape)
						nelems *= d;
					intermediates.emplace_back(nelems, 0.0f);
					ptrs[i] = intermediates.back().data();
				}
			}

			fn_(ptrs.data(), static_cast<int>(tensor_order_.size()));
		}

		[[nodiscard]] const std::vector<std::string> &tensor_order() const {
			return tensor_order_;
		}
	};
} // namespace ten
