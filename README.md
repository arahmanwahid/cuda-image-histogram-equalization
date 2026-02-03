# CUDA Image Histogram

High-performance grayscale histogramming using Shared Memory Privatization.

## 🚀 Results (Tesla T4)
- **Image Size:** 1024 x 1024 (1MP)
- **Status:** Success
- **Verification:** 1,048,576 pixels processed.

## 🛠️ Setup & Usage
```bash
# Clone the clean repository
git clone https://github.com/arahmanwahid/cuda-image-histogram-equalization.git
cd cuda-image-histogram-equalization

# Compile for Tesla T4
nvcc -arch=sm_75 main.cu -o histogram

# Run
./histogram
```

## 🧠 Technical Overview

This project solves the "memory contention" problem using **Shared Memory Privatization**. Instead of having all threads compete for the same 256 global memory bins, each block maintains its own local histogram in high-speed on-chip cache. These are merged into global memory only once per block.
