# Compiler settings
NVCC = nvcc
NVCC_FLAGS = -lcublas
# NVCC_FLAGS = 

# Default compute capability
# ARCH = -arch=sm_86 # for RTX A6000  

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
	@echo "  make <filename>    - Compile specific CUDA file (without .cu extension)"
	@echo "  make run <filename> - Run compiled executable"
	@echo "  make clean        - Remove all compiled files"
	@echo "  make help         - Show this help message"

# Pattern rule for CUDA files
%: %$(CU_EXT)
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $< -o $(BUILD_DIR)/$@
	echo "Compiled $(BUILD_DIR)/$@"

# Run compiled executable
.PHONY: run

run:
	@if [ -f "$(BUILD_DIR)/$(word 2,$(MAKECMDGOALS))" ]; then \
		./$(BUILD_DIR)/$(word 2,$(MAKECMDGOALS)); \
	else \
		echo "Executable $(word 2,$(MAKECMDGOALS)) not found in build directory."; \
		echo "Compiling first..."; \
		$(MAKE) $(word 2,$(MAKECMDGOALS)) && ./$(BUILD_DIR)/$(word 2,$(MAKECMDGOALS)); \
	fi

# Ignore second argument only for the run target
ifeq (run,$(firstword $(MAKECMDGOALS)))
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(RUN_ARGS):;@:)
endif

# Clean build files
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all help clean run
