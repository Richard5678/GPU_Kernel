# Compiler settings
NVCC = nvcc
NVCC_FLAGS =

# Default compute capability
ARCH = -arch=sm_86 # for RTX A6000  

# Source file extension
CU_EXT = .cu

# Build directory
BUILD_DIR = build

# Make sure build directory exists
$(shell mkdir -p $(BUILD_DIR))

# Default target
all: help

# Help message
help:
	@echo "Usage:"
	@echo "  make <filename>     - Compile specific CUDA file (without .cu extension)"
	@echo "  make clean         - Remove all compiled files"
	@echo "  make help          - Show this help message"

# Pattern rule for CUDA files
%: %$(CU_EXT)
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $< -o $(BUILD_DIR)/$@

# Clean build files
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all help clean 