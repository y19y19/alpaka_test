#include "../lib_src/VectorAdd.hpp"
#include <alpaka/alpaka.hpp>
#include <iostream>
#include <vector>
#include <cstdint>

int main()
{
    VectorAdd::VectorAddExecutor VA;
    std::vector<float> vector_A;
    std::vector<float> vector_B;
    for (int i = -10000; i < 10000; i++) {
        vector_A.push_back(i * 0.5);
        vector_B.push_back(i * 1.3 + 5.8);
    }
    VA.Add(vector_A, vector_B);
    std::cout << "Size of output: " << VA.GetOutputSize() << std::endl;
    std::vector<float> C = VA.GetOutput();
    std::cout << "First 5 values: " << C[0] << " " << C[1] << " " << C[2] << " " << C[3] << " " << C[4] << std::endl;

    return 0;
}
