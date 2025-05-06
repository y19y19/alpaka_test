CXX = g++
NVCC = nvcc
GENCODE_CUDA := -gencode arch=compute_75,code=[sm_75,compute_75]
NVCCFLAGS = -O3 -std=c++20 -g --compiler-options -Wall --compiler-options -Wno-deprecated-gpu-targets $(GENCODE_CUDA)
CXXFLAGS = -O3 -std=c++20 -g -Wall

ALPAKA_ROOT = /cvmfs/cms.cern.ch/el8_amd64_gcc12/external/alpaka/1.2.0-b081818336b70095080b83065d50ff0d
BOOST_ROOT=/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/boost/1.80.0-96a02191111b66819e07de179cb25a0e
ALPAKAINCLUDE = -I${ALPAKA_ROOT}/include -I/${BOOST_ROOT}/include
ALPAKA_CUDA = -DALPAKA_ACC_GPU_CUDA_ENABLED -DALPAKA_ACC_GPU_CUDA_ONLY --expt-relaxed-constexpr
ALPAKA_CPU = -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
# Compile main program
all: vectorAdd_cuda vectorAdd_cpu
vectorAdd_cuda: vectorAdd_cuda.o
	$(NVCC) $(NVCCFLAGS) vectorAdd_cuda.o -o vectorAdd_cuda

vectorAdd_cpu: vectorAdd_cpu.o
	$(CXX) $(CXXFLAGS) vectorAdd_cpu.o -o vectorAdd_cpu	

vectorAdd_cuda.o: vectorAdd.cpp
	$(NVCC) -x cu $(NVCCFLAGS) $(ALPAKAINCLUDE) $(ALPAKA_CUDA) -c vectorAdd.cpp -o vectorAdd_cuda.o 

vectorAdd_cpu.o: vectorAdd.cpp
	$(CXX) $(CXXFLAGS) $(ALPAKAINCLUDE) $(ALPAKA_CPU) -c vectorAdd.cpp -o vectorAdd_cpu.o


clean:
	rm -f vectorAdd_cpu.o vectorAdd_cuda.o vectorAdd_cpu vectorAdd_cuda

