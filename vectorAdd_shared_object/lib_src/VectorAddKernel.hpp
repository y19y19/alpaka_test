#ifndef VECTORADDKERNEL_HPP
#define VECTORADDKERNEL_HPP

#include <alpaka/alpaka.hpp>
#include <cstdint>
#include <cstdio>

struct VectorAddKernel
{
    template <typename Acc>
    ALPAKA_FN_ACC void operator()(
        Acc const & acc,
        float* A,
        float* B,
        float* C) const {
            auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
            C[idx] = A[idx] + B[idx];
    }

};



#endif // VECTORADDKERNEL_HPP
