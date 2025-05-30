#include "VectorAdd.hpp"
#include "VectorAddKernel.hpp"

void VectorAdd::VectorAddExecutor::Add(const std::vector<float>& vector_A, const std::vector<float>& vector_B) {


    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);
    Platform platform;
    auto const device = alpaka::getDevByIdx(platform, 0u);
    auto queue = Queue(device);
    size = vector_A.size();
    
    alpaka::Vec<Dim1D, Idx> extent{size};
    auto bufHost_A = alpaka::allocBuf<float, Idx>(devHost, extent);
    auto bufHost_B = alpaka::allocBuf<float, Idx>(devHost, extent);
    auto bufHost_C = alpaka::allocBuf<float, Idx>(devHost, extent);

    h_A = alpaka::getPtrNative(bufHost_A);
    h_B = alpaka::getPtrNative(bufHost_B);
    h_C = alpaka::getPtrNative(bufHost_C);


    for (uint32_t idx = 0; idx < size; idx++) {
        h_A[idx] = vector_A[idx];
        h_B[idx] = vector_B[idx];
    }


    auto bufDev_A = alpaka::allocBuf<float, Idx>(device, extent);
    auto bufDev_B = alpaka::allocBuf<float, Idx>(device, extent);
    auto bufDev_C = alpaka::allocBuf<float, Idx>(device, extent);

    float* d_A = alpaka::getPtrNative(bufDev_A);
    float* d_B = alpaka::getPtrNative(bufDev_B);
    float* d_C = alpaka::getPtrNative(bufDev_C);


    alpaka::memcpy(queue, bufDev_A, bufHost_A, extent);
    alpaka::memcpy(queue, bufDev_B, bufHost_B, extent);

    uint32_t blocksPerGrid = size;
    uint32_t threadsPerBlock = 1;
    uint32_t elementsPerThread = 1;

    auto workDiv = WorkDiv(blocksPerGrid, threadsPerBlock, elementsPerThread);


    VectorAddKernel vectorAddKernel;
    auto taskRunKernel = alpaka::createTaskKernel<Acc1D>(workDiv, vectorAddKernel, d_A, d_B, d_C);


    alpaka::enqueue(queue, taskRunKernel);
    alpaka::memcpy(queue, bufHost_C, bufDev_C, extent);
    alpaka::wait(queue);


    const int counts = size;
    for (int idx = 0; idx < counts; idx++) output.push_back(h_C[idx]);


    return;
}

std::vector<float> VectorAdd::VectorAddExecutor::GetOutput() {
    return output;
}

unsigned int VectorAdd::VectorAddExecutor::GetOutputSize() {
    return  output.size() * sizeof(float);
}
