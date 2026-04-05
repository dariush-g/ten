#pragma once

#include "../src/ten.h"

inline void test_tiling()
{
    auto A = ten::f32({64, 128}, "A");
    auto B = ten::f32({128, 9}, "B");
    auto bias = ten::f32({9}, "bias");

    ten::Builder b;
    auto C = b.matmul(A, B);
    auto D = b.bias_add(C, bias);
    auto E = b.relu(D);

    auto kernel = b.compile();
}
