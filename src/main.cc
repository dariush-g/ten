#include "ten.h"
#include <iostream>

int main()
{
	float A_data[] = {1.0f, 2.0f};
	float B_data[] = {3.0f, 4.0f};
	float bias_data[] = {0.5f};

	auto A = ten::f32({1, 2}, "A");
	auto B = ten::f32({2, 1}, "B");
	auto bias = ten::f32({1}, "bias");

	ten::Builder b;
	auto C = b.matmul(A, B);
	auto D = b.bias_add(C, bias);
	auto E = b.relu(D);

	auto kernel = b.compile();
	kernel({{"A", A_data}, {"B", B_data}, {"bias", bias_data}});

	auto result = kernel.get(E);

	std::cout << result << std::endl;

	return 0;
}
