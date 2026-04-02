#pragma once

#include <string>
#include <vector>

namespace ten {

enum class DType {
	f16,
	bf16,
	f32,
	f64,
	i8,
	i16,
	i32,
	i64,
};

inline int dtype_size(DType dt) {
	switch (dt) {
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
}

inline const char *dtype_str(DType dt) {
	switch (dt) {
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
}

struct TensorLayout {
	std::vector<int> shape;
	DType dtype;
	std::string name = "";

	TensorLayout() = default;

	TensorLayout(std::vector<int> shape, DType dtype = DType::f32,
				 std::string name = "")
		: shape(std::move(shape)), dtype(dtype), name(std::move(name)) {}

	int rank() const { return shape.size(); }
	int dim(int i) const { return shape[i]; }
	int element_size() const { return dtype_size(dtype); }
};

struct TensorBuffer {
	void *data;
	TensorLayout layout;

	TensorBuffer(void *data, TensorLayout layout)
		: data(data), layout(std::move(layout)) {}
};

inline TensorLayout f32(std::vector<int> shape, std::string name = "") {
	return TensorLayout(shape, DType::f32, name);
}

inline TensorLayout f16(std::vector<int> shape, std::string name = "") {
	return TensorLayout(shape, DType::f16, name);
}

inline TensorLayout i8(std::vector<int> shape, std::string name = "") {
	return TensorLayout(shape, DType::i8, name);
}

} // namespace ten