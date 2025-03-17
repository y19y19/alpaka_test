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
all: helloWorld_cuda helloWorld_cpu
helloWorld_cuda: helloWorld_cuda.o
	$(NVCC) $(NVCCFLAGS) helloWorld_cuda.o -o helloWorld_cuda

helloWorld_cpu: helloWorld_cpu.o
	$(CXX) $(CXXFLAGS) helloWorld_cpu.o -o helloWorld_cpu	

helloWorld_cuda.o: helloWorld.cpp
	$(NVCC) -x cu $(NVCCFLAGS) $(ALPAKAINCLUDE) $(ALPAKA_CUDA) -c helloWorld.cpp -o helloWorld_cuda.o 

helloWorld_cpu.o: helloWorld.cpp
	$(CXX) $(CXXFLAGS) $(ALPAKAINCLUDE) $(ALPAKA_CPU) -c helloWorld.cpp -o helloWorld_cpu.o


clean:
	rm -f helloWorld_cpu.o helloWorld_cuda.o helloWorld_cpu helloWorld_cuda

