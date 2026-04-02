#include "ten.h"
#include <iostream>

int main() {
	auto A = ten::f32({512, 256}, "A");
	auto B = ten::f32({256, 128}, "B");
	auto bias = ten::f32({128}, "B");

	ten::Builder b;
	auto C = b.matmul(A, B);
	auto D = b.bias_add(C, bias);
	auto E = b.relu(D);

	auto fn = b.compile();

	return 0;
}