CUDA_ARCH ?= -arch=sm_75
NVCC ?= nvcc
CXXFLAGS ?= -O2 -std=c++17

TARGET := cuda_image_processor
SOURCE := src/image_processing.cu
HEADER := include/image_processing.h

all: $(TARGET)

$(TARGET): $(SOURCE) $(HEADER)
	$(NVCC) $(CUDA_ARCH) $(CXXFLAGS) -Iinclude $(SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)
	rm -rf output/*.pgm

.PHONY: all clean
