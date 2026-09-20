#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

def write_pgm(path, width, height, index):
    with path.open("wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode())
        pixels = bytearray(width * height)
        phase = index * 0.17
        for y in range(height):
            for x in range(width):
                wave = 127.5 + 70.0 * math.sin(x * 0.055 + phase)
                dx = x - width * (0.25 + 0.001 * (index % 20))
                dy = y - height * 0.50
                circle = 55.0 if dx * dx + dy * dy < (min(width, height) * 0.16) ** 2 else 0.0
                pixels[y * width + x] = int(max(0, min(255, wave + circle)))
        f.write(pixels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--output", default="data/input")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.pgm"):
        old.unlink()
    for i in range(args.count):
        write_pgm(out / f"image_{i:04d}.pgm", args.width, args.height, i)
    print(f"Generated {args.count} images at {args.width}x{args.height} in {out}")

if __name__ == "__main__":
    main()
