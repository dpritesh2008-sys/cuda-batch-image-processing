# CUDA Accelerated Batch Image Processing

CUDA project for batch Sobel edge detection on grayscale PGM images.

## Build

```bash
make
```

## Generate a dataset

```bash
python3 scripts/generate_dataset.py --count 200 --width 256 --height 256 --output data/input
```

## Run

```bash
./cuda_image_processor --input data/input --output output --threads 256
```

CLI options: `--input`, `--output`, `--limit`, `--threads`.

The CPU handles PGM file I/O while a custom CUDA kernel processes pixels in parallel using the 3x3 Sobel operator. GPU kernel time is measured with CUDA events.

## Requirements

NVIDIA GPU, CUDA toolkit/nvcc, C++17, and Python 3.

## Project structure

- `src/` CUDA implementation
- `include/` header
- `scripts/` deterministic dataset generator
- `docs/` execution evidence and submission checklist
- `Makefile` build support
- `run.sh` end-to-end runner

## Lessons learned

The project demonstrates batch GPU processing, device memory management, grid-stride execution, CUDA event timing, and practical separation of CPU file I/O from GPU computation.
