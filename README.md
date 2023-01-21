# blurhash algorithm
This project is blurhash implementation in NVIDIA CUDA

## Tests

### Configuratrion

*CPU*: Intel Xeon E5649 2.53 GHz

*GPU*: NVIDIA TESLA K40

Testing samples - 10 jpg images of size up to 1024 x 1024.

### Results
Components:
- 1x1   (extra small)   - GPU: 441ms, CPU: 1830ms,   GPU 4x faster
- 5x5   (typical case)  - GPU: 523ms, CPU: 34977ms,  GPU 67x faster
- 10x10 (detailed blur) - GPU: 798ms, CPU: 139836ms, GPU 175x faster
