#ifndef HELLOWORLDKERNEL_HPP
#define HELLOWORLDKERNEL_HPP

#include <alpaka/alpaka.hpp>
#include <cstdint>
#include <cstdio>

struct HelloWorldKernel {
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const & acc) const {
        uint32_t threadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        printf("Hello, World from alpaka thread %u!\n", threadIdx);
    }
};

#endif // HELLOWORLDKERNEL_HPP
