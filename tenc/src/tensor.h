#pragma once

#include <string>
#include <utility>
#include <vector>
#include <ostream>

namespace ten
{
	enum class DType
	{
		f16,
		bf16,
		f32,
		f64,
		i8,
		i16,
		i32,
		i64,
	};

	inline int dtype_size(const DType dt)
	{
		switch (dt)
		{
		case DType::f16:
			return 2;
		case DType::bf16:
			return 2;
		case DType::f32:
			return 4;
		case DType::f64:
			return 8;
		case DType::i8:
			return 1;
		case DType::i16:
			return 2;
		case DType::i32:
			return 4;
		case DType::i64:
			return 8;
		}
		return 0;
	}

	inline const char* dtype_str(const DType dt)
	{
		switch (dt)
		{
		case DType::f16:
			return "__fp16";
		case DType::bf16:
			return "__bf16";
		case DType::f32:
			return "float";
		case DType::f64:
			return "double";
		case DType::i8:
			return "int8_t";
		case DType::i16:
			return "int16_t";
		case DType::i32:
			return "int32_t";
		case DType::i64:
			return "int64_t";
		}
		return nullptr;
	}

	struct TensorLayout
	{
		std::vector<int> shape;
		DType dtype = DType::f32;
		std::string name;

		TensorLayout() = default;

		explicit TensorLayout(std::vector<int> shape, const DType dtype = DType::f32,
		                      std::string name = "")
			: shape(std::move(shape)), dtype(dtype), name(std::move(name))
		{
		}

		[[nodiscard]] int rank() const { return static_cast<int>(shape.size()); }
		[[nodiscard]] int dim(int i) const { return shape[i]; }
		[[nodiscard]] int element_size() const { return dtype_size(dtype); }
	};

	struct TensorBuffer
	{
		void* data;
		TensorLayout layout;

		TensorBuffer(void* data, TensorLayout layout)
			: data(data), layout(std::move(layout))
		{
		}
	};

	inline TensorLayout f32(std::vector<int> shape, std::string name = "")
	{
		return TensorLayout(std::move(shape), DType::f32, std::move(name));
	}

	inline TensorLayout f16(std::vector<int> shape, std::string name = "")
	{
		return TensorLayout(std::move(shape), DType::f16, std::move(name));
	}

	inline TensorLayout i8(std::vector<int> shape, std::string name = "")
	{
		return TensorLayout(std::move(shape), DType::i8, std::move(name));
	}

	struct TensorView
	{
		const float* data;
		std::vector<int> shape;

		[[nodiscard]] int size() const
		{
			int n = 1;
			for (const int d : shape) n *= d;
			return n;
		}
	};

	inline std::ostream& operator<<(std::ostream& os, const TensorView& view)
	{
		const int size = view.size();
		os << "[";
		for (int i = 0; i < size; i++)
		{
			if (i > 0) os << ", ";
			os << view.data[i];
		}
		os << "]";
		return os;
	}
} // namespace ten

