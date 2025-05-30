#ifndef VECTORADD_H
#define VECTORADD_H

#include "../lib_src/VectorAddKernel.hpp"
#include <alpaka/alpaka.hpp>
#include <iostream>
#include <vector>
#include <cstdint>

using Dim1D = alpaka::DimInt<1u>;
using Idx = uint32_t;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
using Acc1D = alpaka::AccGpuCudaRt<Dim1D, Idx>;
using Platform = alpaka::PlatformCudaRt;
using Queue = alpaka::QueueCudaRtNonBlocking;
#else
using Acc1D = alpaka::AccCpuSerial<Dim1D, Idx>;
using Platform = alpaka::PlatformCpu;
using Queue = alpaka::QueueCpuBlocking;
#endif

using WorkDiv = alpaka::WorkDivMembers<Dim1D, Idx>;


namespace VectorAdd {

    class VectorAddExecutor {
    public: 
        VectorAddExecutor(){};

        unsigned int GetOutputSize();
        std::vector<float> GetOutput();
        void Add(const std::vector<float>& vector_A, const std::vector<float>& vector_B);

    private:
        float* h_A;
        float* h_B;
        float* h_C;
        std::vector<float> output;
        uint32_t size{0};

    };
} //namespace VectorAdd

#endif
 
