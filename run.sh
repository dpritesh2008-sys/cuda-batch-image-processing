#!/usr/bin/env bash
set -euo pipefail

make
python3 scripts/generate_dataset.py --count 200 --width 256 --height 256 --output data/input
./cuda_image_processor --input data/input --output output --threads 256
