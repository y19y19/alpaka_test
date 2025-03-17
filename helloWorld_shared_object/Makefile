# Directories
SRC_DIR = src
OBJ_DIR = obj
LIB_DIR = lib
LIB_SRC_DIR = lib_src
LIB_OBJ_DIR = lib_obj
BIN_DIR = bin

# Files
LIB_SOURCES = $(LIB_SRC_DIR)/helloWorldKernel.cpp
LIB_OBJECTS_CUDA = $(LIB_OBJ_DIR)/helloWorldKernel_cuda.o
LIB_OBJECTS_CPU= $(LIB_OBJ_DIR)/helloWorldKernel_cpu.o

SHARED_LIB_CUDA = libHelloWorld_cuda.so
SHARED_LIB_CPU = libHelloWorld_cpu.so

SOURCES = $(SRC_DIR)/main.cpp
OBJECTS_CPU = $(OBJ_DIR)/main_cpu.o
OBJECTS_CUDA = $(OBJ_DIR)/main_cuda.o

EXECUTABLE_CUDA = $(BIN_DIR)/helloWorld_cuda
EXECUTABLE_CPU = $(BIN_DIR)/helloWorld_cpu


# Compiler and Flags
CXX = g++
NVCC = nvcc
CXXFLAGS = -O2 -std=c++20 -g -Wall
CUDAFLAGS = -x cu -O3 -std=c++20 -g --compiler-options -Wall --compiler-options -Wno-deprecated-gpu-targets --expt-relaxed-constexpr
SOFLAGS_CUDA = --compiler-options -fPIC --cudart shared -dc
SOFLAGS_CPU = -fPIC
GENCODEFLAGS = -gencode arch=compute_75,code=[sm_75,compute_75]
INCLUDEFLAGS = -I/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/alpaka/1.2.0-b081818336b70095080b83065d50ff0d/include -I/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/boost/1.80.0-96a02191111b66819e07de179cb25a0e/include
ALPAKA_CUDA = -DALPAKA_ACC_GPU_CUDA_ENABLED -DALPAKA_ACC_GPU_CUDA_ONLY
ALPAKA_CPU = -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

LDFLAGS_CUDA = -L$(LIB_DIR) -Wl,-rpath=$(LIB_DIR) -lHelloWorld_cuda -L/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/cuda/12.8.0-15bfa86985d46d842bb5ecc3aca6c676/lib64 -lcudart
LDFLAGS_CPU = -L$(LIB_DIR) -Wl,-rpath=$(LIB_DIR) -lHelloWorld_cpu

# Targets
all: $(EXECUTABLE_CUDA) $(EXECUTABLE_CPU)

$(LIB_OBJ_DIR)/helloWorldKernel_cuda.o: $(LIB_SRC_DIR)/helloWorldKernel.cpp
	$(NVCC) $(CUDAFLAGS) $(SOFLAGS_CUDA) $(GENCODEFLAGS) $(INCLUDEFLAGS) $(ALPAKA_CUDA) -c $< -o $@

$(LIB_DIR)/$(SHARED_LIB_CUDA): $(LIB_OBJ_DIR)/helloWorldKernel_cuda.o
	$(NVCC) $(GENCODEFLAGS) -shared -o $@ $^

$(OBJ_DIR)/main_cuda.o: $(SRC_DIR)/main.cpp
	$(NVCC) $(CUDAFLAGS) $(GENCODEFLAGS) $(INCLUDEFLAGS) $(ALPAKA_CUDA) -c $< -o $@

$(EXECUTABLE_CUDA): $(OBJ_DIR)/main_cuda.o $(LIB_DIR)/$(SHARED_LIB_CUDA)
	$(CXX) $(CXXFLAGS)  $(ALPAKA_CUDA) $(LDFLAGS_CUDA) $^ -o $@ 

$(LIB_OBJ_DIR)/helloWorldKernel_cpu.o: $(LIB_SRC_DIR)/helloWorldKernel.cpp
	$(CXX) $(CXXFLAGS) $(SOFLAGS_CPU) $(INCLUDEFLAGS) $(ALPAKA_CPU) -c $< -o $@

$(LIB_DIR)/$(SHARED_LIB_CPU): $(LIB_OBJ_DIR)/helloWorldKernel_cpu.o
	$(CXX) -shared -o $@ $^

$(OBJ_DIR)/main_cpu.o: $(SRC_DIR)/main.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDEFLAGS) $(ALPAKA_CPU) -c $< -o $@

$(EXECUTABLE_CPU): $(OBJ_DIR)/main_cpu.o $(LIB_DIR)/$(SHARED_LIB_CPU)
	$(CXX) $(CXXFLAGS) $(ALPAKA_CPU) $(LDFLAGS_CPU) $^ -o $@ 

clean:
	rm -rf $(OBJ_DIR)/*.o $(LIB_OBJ_DIR)/*.o $(LIB_DIR)/*.so $(BIN_DIR)/*

# Create necessary directories
$(OBJ_DIR) $(LIB_OBJ_DIR) $(LIB_DIR) $(BIN_DIR):
	mkdir -p $@

