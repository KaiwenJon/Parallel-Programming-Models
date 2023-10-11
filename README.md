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
export OMP_NUM_THREADS=4 // if using --openmp
./opencvTest <version> <input_image_path>

// Available versions: --sequential, --openmp, --pthread
// for example: ./opencvTest --openmp ../data/butterfly.jpg
// Result will be saved to ../result.detection.jpg
```

## Execution Time
| Program Version | Execution Time (milliseconds) |
| --------------- | ------------------------ |
| Sequential      | 115                    |
| OpenMP (2 threads) | 58                    |
| Pthread (2 threads) | 60                    |
| OpenMP (4 threads) | 29                    |
| Pthread (4 threads) | 30                    |


![image](https://github.com/KaiwenJon/Parallel-Programming-Models/assets/70893513/10a847a1-5a61-4f9a-b278-58555c387040)


![image](https://github.com/KaiwenJon/Parallel-Programming-Models/assets/70893513/1dcb8294-33c2-49e6-9d9c-bd46060bc2ca)
![image](https://github.com/KaiwenJon/Parallel-Programming-Models/assets/70893513/1817c574-59e1-40b7-b93c-fbf1a693ec2b)
