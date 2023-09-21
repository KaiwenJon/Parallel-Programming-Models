# SIFT Detector
Sequential and OpenMP (Parallel feature extraction) version implemented.

## Algorithm
Using DoG (Difference of Gaussian) to build a scale space. From there, find interest points using NMS (Non-maximum suppression). Draw each SIFT interest point on the image with its feature scale.


## To run the code
```
cd Algo1-SIFT_Detector
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


![image](https://github.com/KaiwenJon/Parallel-Programming-Models/assets/70893513/1dcb8294-33c2-49e6-9d9c-bd46060bc2ca)
![image](https://github.com/KaiwenJon/Parallel-Programming-Models/assets/70893513/1817c574-59e1-40b7-b93c-fbf1a693ec2b)
