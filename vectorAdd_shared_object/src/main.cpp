#include "../lib_src/VectorAddKernel.hpp"
#include <alpaka/alpaka.hpp>
#include <iostream>
#include <vector>
#include <cstdint>

int main()
{
    using Dim1D = alpaka::DimInt<1u>;
    using Idx = uint32_t;

    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    #ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    using Acc1D = alpaka::AccGpuCudaRt<Dim1D, Idx>;
    using Platform = alpaka::PlatformCudaRt;
    using Queue = alpaka::QueueCudaRtNonBlocking;
    #else
    using Acc1D = alpaka::AccCpuSerial<Dim1D, Idx>;
    using Platform = alpaka::PlatformCpu;
    using Queue = alpaka::QueueCpuBlocking;
    #endif

    Platform platform;
    auto const device = alpaka::getDevByIdx(platform, 0u);
    auto queue = Queue(device);

    uint32_t size = 10000;

    alpaka::Vec<Dim1D, Idx> extent{size};
    auto bufHost_A = alpaka::allocBuf<float, Idx>(devHost, extent);
    auto bufHost_B = alpaka::allocBuf<float, Idx>(devHost, extent);
    auto bufHost_C = alpaka::allocBuf<float, Idx>(devHost, extent);

    float* h_A = alpaka::getPtrNative(bufHost_A);
    float* h_B = alpaka::getPtrNative(bufHost_B);
    float* h_C = alpaka::getPtrNative(bufHost_C);

    for (uint32_t idx = 0; idx < size; idx++)
    {
        h_A[idx] = idx;
        h_B[idx] = 1.5;
    }

    std::cout << "Vector A first three elements: "
              << h_A[0] << ", " << h_A[1] << ", " << h_A[2] << std::endl;

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

    using WorkDiv = alpaka::WorkDivMembers<Dim1D, Idx>;
    auto workDiv = WorkDiv(blocksPerGrid, threadsPerBlock, elementsPerThread);

    VectorAddKernel vectorAddKernel;
    auto taskRunKernel = alpaka::createTaskKernel<Acc1D>(workDiv, vectorAddKernel, d_A, d_B, d_C);

    alpaka::enqueue(queue, taskRunKernel);
    alpaka::memcpy(queue, bufHost_C, bufDev_C, extent);
    alpaka::wait(queue);

    std::cout << "Computed vector first three elements: "
              << h_C[0] << ", " << h_C[1] << ", " << h_C[2] << std::endl;

    return 0;
}
