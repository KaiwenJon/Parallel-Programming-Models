# SIFT Detector
Sequential, OpenMP, MPI, and Pthread (Parallel feature extraction) version implemented.

## Algorithm
Using DoG (Difference of Gaussian) to build a scale space. From there, find interest points using NMS (Non-maximum suppression). Draw each SIFT interest point on the image with its feature scale.


## To run the code
```
cd Algo1-SIFT_Detector
mkdir build && cd build
cmake ..
make
export OMP_NUM_THREADS=4 // if using --openmp
./opencvTest <version> <input_image_path>

// Available versions: --sequential, --openmp, --pthread
// for example: ./opencvTest --openmp ../data/butterfly.jpg
// Result will be saved to ../result.detection.jpg

// If using mpi, run this command:
mpiexec -n <number of threads> ./opencvTest --mpi ../data/butterfly.jpg
```

## Execution Time
| Program Version | Execution Time (milliseconds) |
| --------------- | ------------------------ |
| Sequential      | 115                    |
| OpenMP (2 threads) | 58                    |
| Pthread (2 threads) | 60                    |
| MPI (2 threads) | 56                    |
| OpenMP (4 threads) | 29                    |
| Pthread (4 threads) | 30                    |
| MPI (4 threads) | 32                    |


![image](https://github.com/KaiwenJon/Parallel-Programming-Models/assets/70893513/1dcb8294-33c2-49e6-9d9c-bd46060bc2ca)
![image](https://github.com/KaiwenJon/Parallel-Programming-Models/assets/70893513/25e30501-a3e1-4106-bdf6-50c8980d25e0)

![image](https://github.com/KaiwenJon/Parallel-Programming-Models/assets/70893513/050b05a4-73a6-43bc-a63f-ca7b5dd98085)
![image](https://github.com/KaiwenJon/Parallel-Programming-Models/assets/70893513/2e2725f5-ce9a-4a60-84c7-24c8bb4bce0d)

