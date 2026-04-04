#include "mm_ba_rl.h"
#include <iostream>

int main()
{
    test_matmul_bias_add_relu();

    std::cout << "all tests passed" << std::endl;
    return 0;
}
