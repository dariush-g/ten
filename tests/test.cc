#include "mm_ba_rl.h"
#include "tiling.h"
#include <iostream>

int main()
{
    test_matmul_bias_add_relu();
    test_tiling();

    std::cout << "all tests passed" << std::endl;
    return 0;
}
