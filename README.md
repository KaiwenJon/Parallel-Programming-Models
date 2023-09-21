# SIFT Detector
Sequential and OpenMP (Parallel feature extraction) version implemented.

## Algorithm
Using DoG (Difference of Gaussian) to build a scale space. From there, find interest points using NMS (Non-maximum suppression). Draw each SIFT interest point on the image with its feature scale.


## To run the code
```
mkdir build && cd build
cmake ..
make
export OMP_NUM_THREADS=4
./opencvTest <version> <input_image_path>

// Available versions: --sequential, --openmp
// for example: ./opencvTest --openmp ../data/butterfly.jpg
// Result will be saved to ../result.detection.jpg
```

## Execution Time
| Program Version | Execution Time (milliseconds) |
| --------------- | ------------------------ |
| Sequential      | 115                    |
| OpenMP (2 threads) | 58                    |
| OpenMP (4 threads) | 29                    |
