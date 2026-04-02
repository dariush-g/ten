#include "ten.h"
#include <iostream>

int main() {
	auto A = ten::f32({1, 2}, "A");
	auto B = ten::f32({2, 1}, "B");
	auto bias = ten::f32({1}, "bias");

	ten::Builder b;
	auto C = b.matmul(A, B);
	auto D = b.bias_add(C, bias);
	auto E = b.relu(D);

	auto fn = b.compile();

	return 0;
}