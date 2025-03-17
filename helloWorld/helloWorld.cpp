#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <iostream>

struct HelloWorldKernel {
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const & acc) const {
        uint32_t threadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        printf("Hello, World from alpaka thread %u!\n", threadIdx);
    }
};

int main() {

    using Dim1D = alpaka::DimInt<1u>;
    using Idx = uint32_t;

    #ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    using Acc1D = alpaka::AccGpuCudaRt<Dim1D, Idx>;
    // Options
    // - AccCpuOmp2Blocks
    // - AccGpuCudaRt
    // - AccCpuThreads
    // - AccCpuFibers
    // - AccCpuOmp2Threads
    // - AccCpuOmp4
    // - AccCpuTbbBlocks
    // - AccCpuSerial

    //using Device = alpaka::DevCudaRt;
    using Platform = alpaka::PlatformCudaRt;
    using Queue = alpaka::QueueCudaRtNonBlocking;

    #else
    using Acc1D = alpaka::AccCpuSerial<Dim1D, Idx>;
    using Platform = alpaka::PlatformCpu;
    //using Device = alpaka::DevCpu;
    using Queue = alpaka::QueueCpuBlocking;
    #endif
 
    Platform platform;
    auto const device = alpaka::getDevByIdx(platform, 0u);
    auto queue = Queue(device);

    Idx blocksPerGrid = 8;
    Idx threadsPerBlock = 1;
    Idx elementsPerThread = 1;
    using WorkDiv = alpaka::WorkDivMembers<Dim1D, Idx>; 
    auto workDiv = WorkDiv(blocksPerGrid, threadsPerBlock, elementsPerThread);

    HelloWorldKernel helloWorldKernel;
    auto taskRunKernel = alpaka::createTaskKernel<Acc1D>(workDiv, helloWorldKernel);

    alpaka::enqueue(queue, taskRunKernel);

    alpaka::wait(queue);

    return 0;
}
