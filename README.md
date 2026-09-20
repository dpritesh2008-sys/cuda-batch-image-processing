# CUDA Accelerated Batch Image Processing

A CUDA project that performs Sobel edge detection on a batch of grayscale PGM images. CPU code handles file I/O and dataset management; a custom CUDA kernel performs pixel-level image processing in parallel.

## Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit / nvcc
- C++17 compiler
- Python 3

## Build
```bash
make
```

## Generate a large test dataset
```python
python3 scripts/generate_dataset.py --count 200 --width 256 --height 256 --output data/input
```

## Run
```bash
./cuda_image_processor --input data/input --output output --threads 256
```

Or run the complete workflow:
```bash
./run.sh
```

## CLI arguments
- `--input DIR` input directory containing PGM images
- `--output DIR` output directory for edge images
- `--limit N` optional maximum number of images
- `--threads N` CUDA threads per block (1-1024)

## Algorithm
Each image uses the standard 3x3 Sobel masks:
- Gx detects horizontal intensity changes.
- Gy detects vertical intensity changes.
- Edge magnitude is `sqrt(Gx^2 + Gy^2)`, clamped to 0-255.

The CUDA kernel uses a grid-stride loop over the complete batch, so many images and pixels are processed concurrently.

## CUDA design
1. CPU reads all PGM images and packs their pixels into one contiguous host buffer.
2. The buffer is copied to device memory.
3. `SobelBatchKernel` computes the edge magnitude for each pixel.
4. CUDA events measure kernel execution time.
5. Results are copied back and written as PGM images.

The project intentionally keeps file parsing on the CPU and the computationally intensive pixel operation on the GPU.

## Evidence
Run the project on the generated 200-image dataset and save the real terminal output. Record the GPU, CUDA version, dimensions, image count, total pixels, threads per block, and measured kernel time in `docs/execution-evidence-template.md`. Do not invent performance measurements.

## Lessons learned
This project demonstrates CUDA memory allocation and transfers, custom kernel design, grid-stride execution, CUDA event timing, batch processing, and the separation of CPU I/O from GPU computation. A practical challenge is handling image boundaries correctly while keeping the kernel simple and parallel.

## Structure
- `src/image_processing.cu` CUDA kernel and host application
- `include/image_processing.h` declarations
- `scripts/generate_dataset.py` deterministic PGM dataset generator
- `Makefile` build support
- `run.sh` end-to-end runner
- `docs/` evidence and submission checklist
